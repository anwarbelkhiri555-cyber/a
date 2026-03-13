""" IMPORTS """

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import pyrealsense2 as rs
import os


""" GLOBALS """
saved_edges = None
last_capture = None
last_contours = []
last_zones_vis = None
last_zone_masks = None
save_counter = 0
save_folder = "saved_edges"

ROI_X = 120
ROI_Y = 60
ROI_W = 400
ROI_H = 360


""" FUNCTIONS """

def computeEdges(frame):
    edges = cv2.GaussianBlur(frame, (3, 3), sigmaX=1.5, sigmaY=1.5)
    edges = cv2.Canny(edges, threshold1=50, threshold2=200, L2gradient=False)
    return edges


def apply_roi_mask(image, x, y, w, h):
    mask = np.zeros_like(image)

    if len(image.shape) == 2:
        mask[y:y+h, x:x+w] = 255
    else:
        mask[y:y+h, x:x+w] = (255, 255, 255)

    return cv2.bitwise_and(image, mask)


def draw_roi_rectangle(image, x, y, w, h, color=255, thickness=2):
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def preprocess_for_contours(edges_roi):
    kernel_close = np.ones((3, 3), np.uint8)
    kernel_dilate = np.ones((3, 3), np.uint8)

    proc = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    proc = cv2.dilate(proc, kernel_dilate, iterations=1)

    return proc


def contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def select_ring_contours(contours, roi_x, roi_y, roi_w, roi_h):
    candidates = []

    roi_cx = roi_x + roi_w / 2
    roi_cy = roi_y + roi_h / 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.55:
            continue

        center = contour_center(cnt)
        if center is None:
            continue

        cx, cy = center

        dist_to_roi_center = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        if dist_to_roi_center > 80:
            continue

        eq_radius = np.sqrt(area / np.pi)

        candidates.append({
            "cnt": cnt,
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "center": (cx, cy),
            "eq_radius": eq_radius
        })

    if len(candidates) < 3:
        return []

    candidates = sorted(candidates, key=lambda c: c["eq_radius"])

    best_triplet = []
    best_score = None

    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triplet = [candidates[i], candidates[j], candidates[k]]

                centers = np.array([t["center"] for t in triplet], dtype=np.float32)
                mean_center = np.mean(centers, axis=0)
                center_spread = np.mean(np.sqrt(np.sum((centers - mean_center) ** 2, axis=1)))

                radii = [t["eq_radius"] for t in triplet]
                dr1 = radii[1] - radii[0]
                dr2 = radii[2] - radii[1]

                if dr1 < 10 or dr2 < 10:
                    continue

                mean_circularity = np.mean([t["circularity"] for t in triplet])

                score = center_spread - 40 * mean_circularity

                if best_score is None or score < best_score:
                    best_score = score
                    best_triplet = triplet

    if not best_triplet:
        candidates = sorted(candidates, key=lambda c: c["area"], reverse=True)
        return [c["cnt"] for c in candidates[:3]]

    return [t["cnt"] for t in best_triplet]


def build_zone_masks_from_contours(image_shape, contours, roi_x, roi_y, roi_w, roi_h):
    h, w = image_shape[:2]

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255

    if len(contours) != 3: return None

    c1, c2, c3 = contours

    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    m3 = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(m1, [c1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(m2, [c2], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(m3, [c3], -1, 255, thickness=cv2.FILLED)

    y_licht, x_licht = np.where(m3>0)
    global y_min, y_max
    y_min, y_max = np.min(y_licht), np.max(y_licht)
    print( y_min, y_max)
    if len(contours) != 3: return None

    m1 = cv2.bitwise_and(m1, roi_mask)
    m2 = cv2.bitwise_and(m2, roi_mask)
    m3 = cv2.bitwise_and(m3, roi_mask)

    zone1 = m1
    zone2 = cv2.subtract(m2, m1)
    zone3 = cv2.subtract(m3, m2)
    zone4 = cv2.subtract(roi_mask, m3)

    return zone1, zone2, zone3, zone4


def visualize_zones(zone1, zone2, zone3, zone4, contours, roi_x, roi_y, roi_w, roi_h):
    h, w = zone1.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[:] = (70, 70, 70)
    vis[zone3 > 0] = (130, 130, 130)
    vis[zone2 > 0] = (190, 190, 190)
    vis[zone1 > 0] = (255, 255, 255)

    if len(contours) >= 1:
        cv2.drawContours(vis, [contours[0]], -1, (0, 0, 255), 2)
    if len(contours) >= 2:
        cv2.drawContours(vis, [contours[1]], -1, (0, 255, 0), 2)
    if len(contours) >= 3:
        cv2.drawContours(vis, [contours[2]], -1, (255, 0, 0), 2)

    return vis


def get_zone_at_pixel(x, y, zone_masks):
    if zone_masks is None:
        return None

    zone1, zone2, zone3, zone4 = zone_masks

    if y < 0 or y >= zone1.shape[0] or x < 0 or x >= zone1.shape[1]:
        return None

    if zone1[y, x] > 0:
        return 1
    elif zone2[y, x] > 0:
        return 2
    elif zone3[y, x] > 0:
        return 3
    elif zone4[y, x] > 0:
        return 4
    else:
        return None


def analyze_capture(captured_edges):
    processed = preprocess_for_contours(captured_edges)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ring_contours = select_ring_contours(contours, ROI_X, ROI_Y, ROI_W, ROI_H)

    zones_vis = None
    zone_masks = None

    if len(ring_contours) == 3:
        zone_masks = build_zone_masks_from_contours(
            captured_edges.shape,
            ring_contours,
            ROI_X, ROI_Y, ROI_W, ROI_H
        )

        if zone_masks is not None:
            zone1, zone2, zone3, zone4 = zone_masks
            zones_vis = visualize_zones(
                zone1, zone2, zone3, zone4,
                ring_contours,
                ROI_X, ROI_Y, ROI_W, ROI_H
            )

    return ring_contours, zones_vis, zone_masks


def mouse_callback_edges(event, x, y, flags, param):
    global saved_edges, last_capture, save_counter
    global last_contours, last_zones_vis, last_zone_masks

    if event == cv2.EVENT_LBUTTONDOWN:
        if saved_edges is not None:
            filename = os.path.join(save_folder, f"edges_{save_counter:04d}.png")
            cv2.imwrite(filename, saved_edges)
            print("Saved:", filename)

            last_capture = saved_edges.copy()
            last_contours, last_zones_vis, last_zone_masks = analyze_capture(last_capture)

            save_counter += 1


def mouse_callback_zones(event, x, y, flags, param):
    global last_zone_masks

    if event == cv2.EVENT_RBUTTONDOWN:
        zone = get_zone_at_pixel(x, y, last_zone_masks)

        if zone is None:
            print(f"Rechtermuisklik op ({x}, {y}) -> geen geldige zone")
        else:
            print(f"Rechtermuisklik op ({x}, {y}) -> zone {zone}")


""" MAIN """
def init():
    saved_edges = None
    last_capture = None
    last_contours = []
    last_zones_vis = None
    last_zone_masks = None
    save_counter = 0
    save_folder = "saved_edges"
#if __name__ == "__main__":

    os.makedirs(save_folder, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Geen eerste color frame ontvangen van de RealSense.")

        prev_frame = np.asanyarray(color_frame.get_data())
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("RealSense Color")
        cv2.namedWindow("Edges")
        cv2.namedWindow("Last Capture")
        cv2.namedWindow("Zones Screenshot")

        cv2.setMouseCallback("Edges", mouse_callback_edges)
        cv2.setMouseCallback("Zones Screenshot", mouse_callback_zones)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            edges = computeEdges(frame_gray)

            edges_roi = apply_roi_mask(edges, ROI_X, ROI_Y, ROI_W, ROI_H)
            edges_display = draw_roi_rectangle(edges_roi, ROI_X, ROI_Y, ROI_W, ROI_H, color=255, thickness=2)

            saved_edges = edges_roi.copy()

            cv2.imshow("RealSense Color", frame)
            cv2.imshow("Edges", edges_display)

            if last_capture is not None:
                cv2.imshow("Last Capture", last_capture)


            if last_zones_vis is not None:
                cv2.imshow("Zones Screenshot", last_zones_vis)

            k = cv2.waitKey(1)
            if k == 27:
                break

            prev_frame = frame_gray.copy()

            filename = os.path.join(save_folder, f"edges_{save_counter:04d}.png")
            cv2.imwrite(filename, saved_edges)
            print("Saved:", filename)

            last_capture = saved_edges.copy()
            last_contours, last_zones_vis, last_zone_masks = analyze_capture(last_capture)
            #a,b = y_min, y_max
            return last_zone_masks

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
print(init())