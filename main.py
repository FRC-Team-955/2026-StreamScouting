import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from config import (
    FRAME_SKIP,
    MAX_TRAIL,
    PARABOLA_MIN_POINTS,
    PARABOLA_FIT_ERROR,
    SCORE_COOLDOWN_FRAMES,
    SKIP_SECONDS,
    TRAIL_DECAY,
    SCORE_POLYGON_REF_BY_SIDE,
    ATTRIBUTION_MAX_DIST,
    ATTRIBUTION_TIME_TOL,
    MATCH_PERIODS,
)
from tracker import Tracker
from vision import (
    blackout_hole,
    blackout_outside_active,
    check_parabola_score,
    crop_frame,
    detect_apriltags,
    detect_circles,
    fit_conic,
    get_runtime_regions,
    sample_conic_curve,
solve_y
)
from robot_tracker import RobotTracker, detect_robots
from robot_detector import get_yolo_latency_ms

from path_stitcher import PathStitcher



def get_frame_at_index(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def period_for_frame(frame_idx, fps, skip_frames):
    """Return the period name (from MATCH_PERIODS) for a given absolute frame index."""
    elapsed_s = (frame_idx - skip_frames) / max(fps, 1)
    for name, end_s in MATCH_PERIODS:
        if elapsed_s < end_s:
            return name
    return MATCH_PERIODS[-1][0]  # past end of last period — still last period


def period_names():
    """Ordered list of period name strings from config."""
    return [name for name, _ in MATCH_PERIODS]


def attribute_shot(ball_start_frame: int,
                   ball_start_pos: tuple,
                   robot_tracks: dict) -> int | None:
    """Return the robot slot ID most likely to have fired this ball, or None.

    Searches each robot's perma_path for the point whose frame_idx is closest
    to ball_start_frame (within ATTRIBUTION_TIME_TOL), then checks spatial
    distance.  The robot with the best combined score wins.

    perma_path entries are (x, y, frame_idx).
    """
    bx, by = ball_start_pos
    best_slot  = None
    best_dist  = float("inf")

    for slot_id, track in robot_tracks.items():
        if not track.initialized or not track.perma_path:
            continue

        # Binary-search-style: find the perma_path entry closest in time
        # (perma_path is appended in order so frame_idx is monotonically non-decreasing)
        closest_pt = None
        closest_dt = float("inf")
        for pt in track.perma_path:
            px, py, pfidx = pt
            dt = abs(pfidx - ball_start_frame)
            if dt < closest_dt:
                closest_dt = dt
                closest_pt = pt

        if closest_pt is None or closest_dt > ATTRIBUTION_TIME_TOL:
            continue   # no temporally valid sample for this robot

        px, py, _ = closest_pt
        dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
        if dist < ATTRIBUTION_MAX_DIST and dist < best_dist:
            best_dist = dist
            best_slot = slot_id

    return best_slot


def polygon_center(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (np.mean(xs), np.mean(ys))


def adjust_polygon_for_apriltag(video_path, side, frame_240_offset=None):
    if frame_240_offset is None:
        try:
            ref_cap = cv2.VideoCapture("Q18.mp4")
            ref_frame = get_frame_at_index(ref_cap, 240)
            ref_cap.release()

            if ref_frame is not None:
                apriltags = detect_apriltags(ref_frame)
                tag11_ref = [tag for tag in apriltags if tag.get("id") == 11]

                if tag11_ref:
                    ref_frame_h, ref_frame_w = ref_frame.shape[:2]
                    crop_region, _, _, _ = get_runtime_regions(ref_frame_w, ref_frame_h, side)
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region

                    tag11_center = tag11_ref[0]["center"]
                    tag11_in_crop = (tag11_center[0] - crop_x1, tag11_center[1] - crop_y1)

                    ref_polygon = SCORE_POLYGON_REF_BY_SIDE[side]
                    _, _, ref_score_polygon, _ = get_runtime_regions(ref_frame_w, ref_frame_h, side)
                    runtime_center = polygon_center(ref_score_polygon)

                    frame_240_offset = (
                        runtime_center[0] - tag11_in_crop[0],
                        runtime_center[1] - tag11_in_crop[1]
                    )
                    print(f"[AprilTag] Reference offset from Q18.mp4: {frame_240_offset}")
        except Exception as e:
            print(f"[AprilTag] Could not calculate reference offset: {e}")
            return None

    if frame_240_offset is None:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        frame = get_frame_at_index(cap, 240)
        cap.release()

        if frame is None:
            print("[AprilTag] Could not read frame 240 from video")
            return None

        apriltags = detect_apriltags(frame)
        tag11_detections = [tag for tag in apriltags if tag.get("id") == 11]

        if not tag11_detections:
            print("[AprilTag] AprilTag 11 not found at frame 240")
            return None

        frame_h, frame_w = frame.shape[:2]
        crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region

        tag11_center = tag11_detections[0]["center"]
        tag11_in_crop = (tag11_center[0] - crop_x1, tag11_center[1] - crop_y1)
        target_polygon_center = (
            tag11_in_crop[0] + frame_240_offset[0],
            tag11_in_crop[1] + frame_240_offset[1]
        )

        current_center = polygon_center(score_polygon)
        translation = (
            target_polygon_center[0] - current_center[0],
            target_polygon_center[1] - current_center[1]
        )

        ref_polygon = SCORE_POLYGON_REF_BY_SIDE[side]
        ref_w, ref_h = 1366, 768
        sx = frame_w / ref_w
        sy = frame_h / ref_h

        adjusted_polygon = [
            (int(x * sx + translation[0]), int(y * sy + translation[1]))
            for x, y in ref_polygon
        ]

        print(f"[AprilTag] Found AprilTag 11 at frame 240 center: {tag11_center}")
        print(f"[AprilTag] Adjusted polygon by translation: ({translation[0]:.1f}, {translation[1]:.1f})")

        return adjusted_polygon

    except Exception as e:
        print(f"[AprilTag] Error adjusting polygon: {e}")
        return None

def run(video_path, side, frame_skip=FRAME_SKIP, max_stale_frames=0):
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Could not read input video dimensions")

    crop_region, hole_region, score_polygon, active_region = get_runtime_regions(frame_w, frame_h, side)

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(SKIP_SECONDS * fps)

    print(
        f"[regions] src={frame_w}x{frame_h} crop={crop_region} "
        f"hole={hole_region} score={score_polygon} active={active_region}"
    )

    tracker    = Tracker()
    trails     = defaultdict(list)
    t_alphas   = defaultdict(list)
    full_trails = defaultdict(list)  # unbounded, no decay — used for scored snapshots
    first_seen = {}
    origins    = {}
    fit_cache  = {}  # oid -> (trail_len, params, err, x_min, x_max)

    score                   = 0
    last_score_frame        = -SCORE_COOLDOWN_FRAMES
    last_score_frame_per_id = defaultdict(lambda: -10)
    scored_track_ids        = set()
    scored_curves           = {}   # oid -> (trail_pts, curve_pts), drawn permanently

    # Per-robot scores: {slot_id: {"total": int, <period_name>: int, ...}}
    _pnames = period_names()
    robot_scores = {i: {"total": 0, **{p: 0 for p in _pnames}}
                    for i in range(6)}
    # {canon_oid -> slot_id} so we can attribute post-hoc stitched shots
    shot_robot_map: dict = {}

    stitcher        = PathStitcher()
    fps_window      = 30
    frame_times     = []
    display_fps     = 0.0
    last_fps_update = time.time()
    frame_idx       = 0
    crop_x, crop_y  = crop_region[0], crop_region[1]

    while True:
        for _ in range(max(0, frame_skip - 1)):
            cap.grab()
            frame_idx += 1

        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx < skip_frames:
            continue

        t_start = time.perf_counter()

        frame = crop_frame(raw_frame, crop_region)
        t_crop = time.perf_counter()

        circles = detect_circles(frame, hole_region, active_region)
        t_circles = time.perf_counter()

        objects = tracker.update([(x, y) for (x, y, r) in circles])
        t_track = time.perf_counter()

        t_before_robots = time.perf_counter()
        robot_dets_full = detect_robots(raw_frame, alliance="both", max_stale_frames=max_stale_frames)
        robot_dets = [
            (cx - crop_x, cy - crop_y, w, h, alliance,
             x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y)
            for (cx, cy, w, h, alliance, x1, y1, x2, y2) in robot_dets_full
        ]
        t_robots = time.perf_counter()
        yolo_waited = max_stale_frames > 0 and (t_robots - t_before_robots) > 0.002

        live, dormant = robot_tracker.update(robot_dets, crop_frame=frame)
        frame = robot_tracker.draw(frame)
        t_rtrack = time.perf_counter()

        for oid, (x, y) in objects.items():
            track = tracker.tracks.get(oid)
            if track and track.ghost_count == 0:
                if oid not in first_seen:
                    first_seen[oid] = (x, y)
                trails[oid].append((x, y))
                t_alphas[oid].append(1.0)
                full_trails[oid].append((x, y))
                if len(trails[oid]) > MAX_TRAIL:
                    trails[oid].pop(0)
                    t_alphas[oid].pop(0)

        # Re-identification: tracker resurrected a dead ID — full_trail is
        # already on the right key so nothing extra needed; but if a *new*
        # id was about to be created and we merged it back, the trail is
        # already continuous. No-op here; the tracker handles it in-place.

        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps = zip(*combined)
                trails[oid]   = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)
                fit_cache.pop(oid, None)

        # ── live trail stitching (parabolic + velocity coherence) ──────────
        stitcher.update(trails, t_alphas, full_trails, tracker, frame_idx)

        # ── trail decay ────────────────────────────────────────────────────
        for oid in list(t_alphas.keys()):
            t_alphas[oid] = [a * TRAIL_DECAY for a in t_alphas[oid]]
            combined = [(p, a) for p, a in zip(trails[oid], t_alphas[oid]) if a > 0.05]
            if combined:
                pts, alps = zip(*combined)
                trails[oid] = list(pts)
                t_alphas[oid] = list(alps)
            else:
                trails.pop(oid, None)
                t_alphas.pop(oid, None)
                # also evict fit cache inside stitcher
                stitcher._fit_cache.pop(oid, None)

        # ── score check (now sees stitched trails) ─────────────────────────
        for oid, pts in trails.items():
            # Resolve canonical ID in case this oid was a continuation
            canon_oid = stitcher.remap.get(oid, oid)
            track = tracker.tracks.get(oid)
            track_lost = bool(track and track.ghost_count == 1)
            if check_parabola_score(
                    canon_oid, pts, frame_idx, last_score_frame_per_id,
                    last_score_frame, score_polygon, scored_track_ids,
                    track_lost=track_lost,
            ):
                score += 1
                last_score_frame_per_id[canon_oid] = frame_idx
                last_score_frame = frame_idx
                scored_track_ids.add(canon_oid)
                if canon_oid not in origins and canon_oid in first_seen:
                    origins[canon_oid] = first_seen[canon_oid]

                if canon_oid not in scored_curves:
                    trail_snap = list(full_trails.get(oid, []))
                    scored_curves[canon_oid] = (trail_snap, [])

                # ── Attribute this shot to the nearest robot ──────────────
                ball_start_pos   = full_trails[oid][0] if full_trails.get(oid) else pts[0]
                ball_start_frame = frame_idx - len(pts) + 1  # approx launch frame
                slot_id = attribute_shot(ball_start_frame, ball_start_pos,
                                         robot_tracker.tracks)
                if slot_id is not None:
                    period = period_for_frame(ball_start_frame, fps, skip_frames)
                    robot_scores[slot_id]["total"]  += 1
                    robot_scores[slot_id][period]   += 1
                    shot_robot_map[canon_oid] = slot_id
                    print(f"  [SCORE]  ID {canon_oid} @ frame {frame_idx} -> total: {score}"
                          f"  robot slot {slot_id} ({period})")
                else:
                    print(f"  [SCORE]  ID {canon_oid} @ frame {frame_idx} -> total: {score}"
                          f"  (no robot attribution)")

        # ── post-hoc scored-curve stitching (upgraded, replaces old loop) ──
        scored_curves = stitcher.stitch_scored_curves(scored_curves)

        t_score = time.perf_counter()

        vis = frame.copy()
        cv2.polylines(vis, [np.array(score_polygon, dtype=np.int32)], True, (0, 255, 0), 2)

        for (x, y, r) in circles:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        for oid in trails:
            pts  = trails[oid]
            alps = t_alphas[oid]
            for i in range(1, len(pts)):
                a     = alps[i - 1]
                color = (0, int(255 * a), int(255 * (1 - a)))
                cv2.line(vis, pts[i - 1], pts[i], color, 2)

        # --- parabola fitting disabled ---
        # vis_h = vis.shape[0]
        # vis_w = vis.shape[1]
        # for oid, pts in trails.items():
        #     if len(pts) >= PARABOLA_MIN_POINTS:
        #         trail_len = len(pts)
        #         if oid not in fit_cache or fit_cache[oid][0] != trail_len:
        #             xs = np.array([p[0] for p in pts])
        #             ys = np.array([p[1] for p in pts])
        #             if max(xs.max() - xs.min(), ys.max() - ys.min()) < 20:
        #                 continue
        #             params, err = fit_conic(xs, ys)
        #             a, b, c, theta = params
        #             curve_pts = sample_conic_curve(params, xs, ys, vis_w, vis_h)
        #             fit_cache[oid] = (trail_len, params, err, float(xs.min()), float(xs.max()), curve_pts)
        #         _, params, err, x_min, x_max, curve_pts = fit_cache[oid]
        #         col = (0, 200, 255) if err < PARABOLA_FIT_ERROR else (80, 80, 80)
        #         if len(curve_pts) >= 5:
        #             cv2.polylines(vis, [np.array(curve_pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)

        for tid, track in tracker.tracks.items():
            if track.ghost_count > 0:
                px, py = track.position()
                alpha  = 1.0 - track.ghost_count / 4
                color  = (int(100 * alpha), int(100 * alpha), int(200 * alpha))
                cv2.circle(vis, (px, py), 6, color, 1)

        for oid, (x, y) in objects.items():
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(oid), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        t_draw = time.perf_counter()

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > fps_window:
            frame_times.pop(0)
        now = time.time()
        if now - last_fps_update > 0.5 and frame_times:
            display_fps     = 1.0 / (sum(frame_times) / len(frame_times))
            last_fps_update = now

        print(
            f"[time] crop={t_crop-t_start:.3f} "
            f"circles={t_circles-t_crop:.3f} "
            f"track={t_track-t_circles:.3f} "
            f"robots={t_robots-t_before_robots:.3f}{'*' if yolo_waited else ''} "
            f"rtrack={t_rtrack-t_robots:.3f} "
            f"score={t_score-t_rtrack:.3f} "
            f"draw={t_draw-t_score:.3f} "
            f"total={t_end-t_start:.3f}"
        )

        # ── Draw permanently retained scored-shot trails ───────────────────
        for shot_idx, (oid, (trail_pts, cpts)) in enumerate(scored_curves.items(), start=1):
            # Color the trail by which robot shot it (slot color) or orange if unattributed
            r_slot = shot_robot_map.get(oid)
            from robot_tracker import _SLOT_COLORS
            shot_color = _SLOT_COLORS[r_slot % len(_SLOT_COLORS)] if r_slot is not None \
                         else (0, 165, 255)
            for i in range(1, len(trail_pts)):
                cv2.line(vis, trail_pts[i - 1], trail_pts[i], shot_color, 2, cv2.LINE_AA)
            mid = trail_pts[len(trail_pts) // 2]
            cv2.putText(vis, f"#{shot_idx}", (mid[0] + 4, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, shot_color, 1, cv2.LINE_AA)

        # ── Robot score labels next to each live robot box ─────────────────
        from robot_tracker import _SLOT_COLORS
        for slot_id, track in robot_tracker.tracks.items():
            if not track.initialized:
                continue
            rs = robot_scores[slot_id]
            alliance_word = {"red": "RED", "blue": "BLU", "unknown": "UNK"}.get(
                track.alliance, "UNK")
            same_alliance = [sid for sid, t in robot_tracker.tracks.items()
                             if t.alliance == track.alliance and sid <= slot_id]
            alliance_num  = len(same_alliance)
            robot_label   = f"{alliance_word} {alliance_num}"
            # Build period breakdown dynamically e.g. "A:2 T:1 E:0"
            period_parts  = " ".join(f"{n[0].upper()}:{rs[n]}" for n, _ in MATCH_PERIODS)
            score_label   = f"{rs['total']}pt  {period_parts}"

            slot_color = _SLOT_COLORS[slot_id % len(_SLOT_COLORS)]
            if track.last_box is not None and track.ghost_count == 0:
                bx2 = track.last_box[2]
                by1 = track.last_box[1]
            else:
                px, py = track.position()
                bx2, by1 = px + track.w // 2, py - track.h // 2

            cv2.putText(vis, robot_label, (bx2 + 4, by1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, slot_color, 1, cv2.LINE_AA)
            cv2.putText(vis, score_label, (bx2 + 4, by1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_color, 1, cv2.LINE_AA)

        # ── Current game period indicator ──────────────────────────────────
        current_period = period_for_frame(frame_idx, fps, skip_frames)
        # Cycle through distinct colors for however many periods exist
        _period_colors = [(0,255,255),(0,255,0),(0,140,255),(255,200,0),(200,0,255)]
        _pidx = next((i for i, (n,_) in enumerate(MATCH_PERIODS) if n == current_period), 0)
        period_color = _period_colors[_pidx % len(_period_colors)]
        cv2.putText(vis, current_period.upper(), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, period_color, 2, cv2.LINE_AA)

        # ── Top-right HUD ──────────────────────────────────────────────────
        cv2.putText(vis, f"Score: {score}", (460, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f"{display_fps:.1f}/{(60 / max(1, frame_skip)):.0f} fps  tracks: {len(tracker.tracks)}",
                    (460, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        yolo_ms    = get_yolo_latency_ms()
        yolo_color = (0, 255, 180) if yolo_ms < 80 else (0, 140, 255)
        try:
            from robot_tracker import _USE_GPU as _rg
        except ImportError:
            _rg = False
        wait_tag = "  WAIT" if yolo_waited else ""
        cv2.putText(vis, f"YOLO {yolo_ms:.0f} ms  Motion: {'GPU' if _rg else 'CPU'}{wait_tag}",
                    (460, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, yolo_color, 1)

        # ── Live scoreboard table ──────────────────────────────────────────
        # Drawn as a semi-transparent dark panel at bottom-left so it's
        # always readable regardless of field content.
        _pnames     = period_names()
        attributed  = sum(rs["total"] for rs in robot_scores.values())
        t_factor    = attributed / score if score > 0 else 0.0
        col_w       = 44   # px per period column
        row_h       = 16
        n_cols      = 2 + len(_pnames) + 1   # robot | total | periods... | scaled
        table_w     = 88 + col_w * (len(_pnames) + 1)
        n_rows      = 7   # header + 6 robots
        table_h     = row_h * (n_rows + 1) + 8
        tx, ty      = 4, vis.shape[0] - table_h - 4

        # Dark background panel
        overlay = vis.copy()
        cv2.rectangle(overlay, (tx - 2, ty - 2),
                      (tx + table_w, ty + table_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        # Header row
        hdr_y = ty + row_h
        cv2.putText(vis, "ROBOT", (tx + 2, hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(vis, "TOT", (tx + 90, hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        for pi, pname in enumerate(_pnames):
            cv2.putText(vis, pname[:3].upper(), (tx + 90 + col_w*(pi+1), hdr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(vis, "EST", (tx + 90 + col_w*(len(_pnames)+1), hdr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        # Divider
        div_y = ty + row_h + 3
        cv2.line(vis, (tx, div_y), (tx + table_w, div_y), (100,100,100), 1)

        for row, (slot_id, rs) in enumerate(robot_scores.items()):
            ry = ty + row_h * (row + 2) + 4
            rtrack   = robot_tracker.tracks[slot_id]
            aw       = {"red":"RED","blue":"BLU","unknown":"UNK"}.get(rtrack.alliance,"UNK")
            same_al  = [sid for sid,t in robot_tracker.tracks.items()
                        if t.alliance == rtrack.alliance and sid <= slot_id]
            anum     = len(same_al)
            row_label = f"{aw}{anum}"
            slot_col  = _SLOT_COLORS[slot_id % len(_SLOT_COLORS)]

            cv2.putText(vis, row_label, (tx + 2, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            cv2.putText(vis, str(rs["total"]), (tx + 90, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            for pi, pname in enumerate(_pnames):
                cv2.putText(vis, str(rs[pname]),
                            (tx + 90 + col_w*(pi+1), ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)
            scaled = rs["total"] / t_factor if t_factor > 0 else 0.0
            cv2.putText(vis, f"{scaled:.1f}",
                        (tx + 90 + col_w*(len(_pnames)+1), ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, slot_col, 1)

        # Footer: tracking factor
        foot_y = ty + table_h - 2
        cv2.putText(vis, f"trk={t_factor:.2f}  attr={attributed}/{score}",
                    (tx + 2, foot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160,160,160), 1)

        cv2.imshow("tracking", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── End-of-match summary ───────────────────────────────────────────────
    attributed_total = sum(rs["total"] for rs in robot_scores.values())
    # Tracking error factor: what fraction of scored balls we could attribute
    # to a specific robot.  1.0 = perfect attribution, <1.0 = some shots lost.
    tracking_factor = attributed_total / score if score > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  MATCH SCORE (ball tracker): {score}")
    print(f"  Attributed shots:           {attributed_total}")
    print(f"  Tracking error factor:      {tracking_factor:.3f}  "
          f"({'perfect' if tracking_factor == 1.0 else 'some shots unattributed'})")
    print("-" * 60)
    print(f"  {'Robot':<12} {'Total':>5}  {'Auto':>5}  {'Teleop':>6}  {'Endgame':>7}  {'Scaled':>7}")
    for slot_id, rs in robot_scores.items():
        track = robot_tracker.tracks[slot_id]
        alliance_word = {"red": "RED", "blue": "BLU", "unknown": "UNK"}.get(
            track.alliance, "UNK")
        same_alliance = [sid for sid, t in robot_tracker.tracks.items()
                         if t.alliance == track.alliance and sid <= slot_id]
        alliance_num  = len(same_alliance)
        label = f"{alliance_word} {alliance_num} (s{slot_id})"
        # Scale each robot's score by the tracking factor — gives best estimate
        # of true contribution accounting for missed attributions
        scaled = rs["total"] / tracking_factor if tracking_factor > 0 else 0.0
        print(f"  {label:<12} {rs['total']:>5}  {rs['auto']:>5}  {rs['teleop']:>6}"
              f"  {rs['endgame']:>7}  {scaled:>7.2f}")
    print("=" * 60 + "\n")

    return score



robot_tracker = RobotTracker()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track")
    parser.add_argument("--side", type=str, choices=["red", "blue"], default="red")
    parser.add_argument("--frame-drop", type=int)
    parser.add_argument("--max-stale-frames", type=int, default=0,
                        help="Max frames a YOLO result can be stale before it's suppressed (0 = unlimited)")
    parser.add_argument("--video-file", type=str)
    args = parser.parse_args()

    cv2.setNumThreads(os.cpu_count() or 1)

    frame_skip = args.frame_drop if args.frame_drop is not None else FRAME_SKIP

    print("Initializing...")
    sc = run(args.video_file, args.side, frame_skip=frame_skip, max_stale_frames=args.max_stale_frames)
    print(f"Final score: {sc}")