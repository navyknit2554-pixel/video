import streamlit as st
import os
import tempfile
import numpy as np
import json
import subprocess
from pathlib import Path

st.set_page_config(
    page_title="패스파인더 영상 편집기",
    page_icon="🎬",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 800; }
    .main-header p  { opacity: .8; margin: .4rem 0 0; }
    .feature-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    }
    .feature-card h4 { margin: 0 0 .5rem; color: #1e293b; }
    .step-badge {
        background: #0f3460; color: white;
        border-radius: 50%; width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: .85rem; font-weight: bold; margin-right: .5rem;
    }
    .status-processing { color: #f59e0b; font-weight: 600; }
    .status-done       { color: #10b981; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🎬 패스파인더 영상 편집기</h1>
  <p>자동 컷편집 · 자막 · 배경음 · 이미지 삽입 · 인물 트래킹</p>
</div>
""", unsafe_allow_html=True)

# ── Session state 초기화 ────────────────────────────────────────────────────
for k, v in {
    "subtitles": [],           # [{start, end, text}]
    "images": [],              # [{start, end, path, position, opacity}]
    "processing": False,
    "output_path": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── helpers ──────────────────────────────────────────────────────────────────
def sec_to_ts(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def run_ffmpeg(cmd, desc="ffmpeg"):
    """Run ffmpeg quietly, return (ok, stderr)."""
    result = subprocess.run(
        ["ffmpeg", "-y"] + cmd,
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stderr

def get_video_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", path],
        capture_output=True, text=True
    )
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0

def get_audio_rms_per_chunk(video_path, chunk_ms=100):
    """Extract per-chunk RMS dB using ffmpeg → raw pcm → numpy."""
    try:
        import librosa, soundfile as sf
        tmp_wav = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ar", "16000", "-ac", "1", tmp_wav],
            capture_output=True
        )
        y, sr = librosa.load(tmp_wav, sr=16000, mono=True)
        os.unlink(tmp_wav)
        chunk_samples = int(sr * chunk_ms / 1000)
        chunks = [y[i:i+chunk_samples] for i in range(0, len(y), chunk_samples)]
        rms_db = []
        for c in chunks:
            rms = np.sqrt(np.mean(c**2) + 1e-10)
            rms_db.append(20 * np.log10(rms))
        return rms_db, chunk_ms / 1000.0
    except Exception as e:
        st.error(f"오디오 분석 오류: {e}")
        return [], 0.1

def detect_silent_intervals(rms_db, chunk_dur, threshold_db, min_silence_sec=0.5):
    """Return list of (start, end) silent intervals."""
    silent = []
    start = None
    for i, db in enumerate(rms_db):
        t = i * chunk_dur
        if db < threshold_db:
            if start is None:
                start = t
        else:
            if start is not None and (t - start) >= min_silence_sec:
                silent.append((start, t))
            start = None
    if start is not None and (len(rms_db) * chunk_dur - start) >= min_silence_sec:
        silent.append((start, len(rms_db) * chunk_dur))
    return silent

def silence_to_keep(duration, silent_intervals):
    """Invert silent intervals → keep intervals."""
    keep, prev = [], 0.0
    for s, e in silent_intervals:
        if s > prev + 0.05:
            keep.append((prev, s))
        prev = e
    if prev < duration - 0.05:
        keep.append((prev, duration))
    return keep

def build_concat_filter(keep_intervals, video_path, output_path):
    """Use ffmpeg complex filter to cut & concat kept segments."""
    parts = []
    filter_v, filter_a = [], []
    for i, (s, e) in enumerate(keep_intervals):
        dur = e - s
        parts += ["-ss", str(s), "-t", str(dur), "-i", video_path]
        filter_v.append(f"[{i}:v]")
        filter_a.append(f"[{i}:a]")
    n = len(keep_intervals)
    filter_complex = (
        "".join(filter_v) + f"concat=n={n}:v=1:a=0[outv];"
        + "".join(filter_a) + f"concat=n={n}:v=0:a=1[outa]"
    )
    cmd = parts + [
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        output_path
    ]
    return cmd

def detect_person_boxes(video_path, sample_fps=2):
    """Detect person bounding boxes using mediapipe pose."""
    try:
        import mediapipe as mp_lib
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, int(fps / sample_fps))
        boxes = {}
        mp_pose = mp_lib.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False,
                             model_complexity=0,
                             min_detection_confidence=0.5)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    xs = [l.x for l in lms if l.visibility > 0.3]
                    ys = [l.y for l in lms if l.visibility > 0.3]
                    if xs and ys:
                        boxes[frame_idx] = (
                            max(0, min(xs) - 0.1),
                            max(0, min(ys) - 0.1),
                            min(1, max(xs) + 0.1),
                            min(1, max(ys) + 0.1)
                        )
            frame_idx += 1
        cap.release()
        pose.close()
        return boxes, w, h, fps, total
    except Exception as e:
        return {}, 0, 0, 30, 0

def apply_tracking(input_path, output_path, track_width_ratio=0.6,
                   smooth_frames=15, progress_cb=None):
    """Crop video to follow detected person."""
    import cv2
    boxes, W, H, fps, total = detect_person_boxes(input_path, sample_fps=3)
    if not boxes:
        return False, "인물을 감지하지 못했습니다."
    # Interpolate boxes for every frame
    all_frames = sorted(boxes.keys())
    cx_arr = np.array([0.5*(boxes[f][0]+boxes[f][2]) for f in all_frames])
    cy_arr = np.array([0.5*(boxes[f][1]+boxes[f][3]) for f in all_frames])
    full_cx = np.interp(range(total), all_frames, cx_arr)
    full_cy = np.interp(range(total), all_frames, cy_arr)
    # Smooth
    kernel = np.ones(smooth_frames) / smooth_frames
    full_cx = np.convolve(full_cx, kernel, mode='same')
    full_cy = np.convolve(full_cy, kernel, mode='same')

    crop_w = int(W * track_width_ratio)
    crop_h = int(crop_w * H / W)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_no_audio = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(tmp_no_audio, fourcc, fps, (crop_w, crop_h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cx = int(np.clip(full_cx[idx] * W, crop_w//2, W - crop_w//2))
        cy = int(np.clip(full_cy[idx] * H, crop_h//2, H - crop_h//2))
        x1, y1 = cx - crop_w//2, cy - crop_h//2
        x2, y2 = x1 + crop_w, y1 + crop_h
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        cropped = frame[y1:y2, x1:x2]
        if cropped.shape[:2] != (crop_h, crop_w):
            cropped = cv2.resize(cropped, (crop_w, crop_h))
        out.write(cropped)
        idx += 1
        if progress_cb and idx % 30 == 0:
            progress_cb(idx / max(total, 1))
    cap.release()
    out.release()
    # Re-mux audio
    ok, err = run_ffmpeg([
        "-i", tmp_no_audio, "-i", input_path,
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path
    ])
    os.unlink(tmp_no_audio)
    return ok, err

def add_subtitles_ffmpeg(input_path, output_path, subtitles):
    """Burn subtitles using ffmpeg drawtext filter."""
    filter_parts = []
    for sub in subtitles:
        s   = float(sub["start"])
        e   = float(sub["end"])
        txt = sub["text"].replace("'", "\\'").replace(":", "\\:")
        filter_parts.append(
            f"drawtext=text='{txt}'"
            f":fontsize=36:fontcolor=white:borderw=3:bordercolor=black"
            f":x=(w-text_w)/2:y=h-80"
            f":enable='between(t,{s},{e})'"
        )
    vf = ",".join(filter_parts) if filter_parts else "null"
    ok, err = run_ffmpeg([
        "-i", input_path, "-vf", vf,
        "-c:a", "copy", output_path
    ])
    return ok, err

def add_bgm_ffmpeg(input_path, output_path, bgm_path, bgm_volume=0.3):
    """Mix background music into video."""
    ok, err = run_ffmpeg([
        "-i", input_path, "-i", bgm_path,
        "-filter_complex",
        f"[1:a]volume={bgm_volume},aloop=loop=-1:size=2e+09[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", output_path
    ])
    return ok, err

def overlay_images_ffmpeg(input_path, output_path, image_overlays):
    """Overlay images onto video at specified times."""
    if not image_overlays:
        return True, ""
    # Build filter_complex chain
    inputs = ["-i", input_path]
    for ov in image_overlays:
        inputs += ["-i", ov["path"]]
    n = len(image_overlays)
    # Build overlay chain
    prev = "0:v"
    filter_parts = []
    for i, ov in enumerate(image_overlays):
        idx   = i + 1
        s, e  = ov["start"], ov["end"]
        pos   = ov.get("position", "center")
        opacity = ov.get("opacity", 1.0)
        pos_map = {
            "center":       "(W-w)/2:(H-h)/2",
            "top-left":     "10:10",
            "top-right":    "W-w-10:10",
            "bottom-left":  "10:H-h-10",
            "bottom-right": "W-w-10:H-h-10",
        }
        xy = pos_map.get(pos, "(W-w)/2:(H-h)/2")
        out_label = f"v{i}"
        filter_parts.append(
            f"[{prev}][{idx}:v]"
            f"overlay={xy}:enable='between(t,{s},{e})'[{out_label}]"
        )
        prev = out_label
    filter_complex = ";".join(filter_parts)
    cmd = inputs + [
        "-filter_complex", filter_complex,
        "-map", f"[{prev}]", "-map", "0:a",
        "-c:a", "copy", output_path
    ]
    ok, err = run_ffmpeg(cmd)
    return ok, err

# ────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Upload & settings ─────────────────────────────────────────────────
with col_left:

    # 1. 영상 업로드
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>1</span> 영상 업로드", unsafe_allow_html=True)
    video_file = st.file_uploader("MP4, MOV, AVI, MKV", type=["mp4","mov","avi","mkv"])
    if video_file:
        tmp_video = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(video_file.name).suffix)
        tmp_video.write(video_file.read()); tmp_video.flush()
        VIDEO_PATH = tmp_video.name
        st.session_state["video_path"] = VIDEO_PATH
        dur = get_video_duration(VIDEO_PATH)
        st.session_state["duration"] = dur
        st.video(video_file)
        st.caption(f"길이: {sec_to_ts(dur)}")
    st.markdown('</div>', unsafe_allow_html=True)

    if "video_path" not in st.session_state:
        st.info("영상을 먼저 업로드해 주세요.")
        st.stop()

    duration = st.session_state.get("duration", 0)

    # 2. 자동 컷편집
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>2</span> 🔇 자동 컷편집 (무음 제거)", unsafe_allow_html=True)
    use_cut = st.toggle("자동 컷편집 사용", value=False)
    if use_cut:
        db_thresh    = st.slider("무음 기준 (dB)", -60, -10, -35)
        min_silence  = st.slider("최소 무음 길이 (초)", 0.2, 3.0, 0.5, 0.1)
        padding_ms   = st.slider("컷 앞뒤 여유 (ms)", 0, 500, 100, 50)
        st.caption(f"기준: {db_thresh} dB 이하 구간을 제거합니다.")
        st.session_state["cut_settings"] = {
            "enabled": True, "db_thresh": db_thresh,
            "min_silence": min_silence, "padding": padding_ms / 1000
        }
    else:
        st.session_state["cut_settings"] = {"enabled": False}
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. 자막
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>3</span> 💬 자막 추가", unsafe_allow_html=True)
    use_sub = st.toggle("자막 사용", value=False)
    if use_sub:
        with st.expander("자막 항목 추가", expanded=True):
            c1, c2, c3 = st.columns([1,1,2])
            sub_start = c1.number_input("시작(초)", 0.0, duration, 0.0, 0.5, key="ss")
            sub_end   = c2.number_input("끝(초)",   0.0, duration, 3.0, 0.5, key="se")
            sub_text  = c3.text_input("자막 내용", key="st")
            if st.button("➕ 자막 추가"):
                if sub_text.strip():
                    st.session_state["subtitles"].append(
                        {"start": sub_start, "end": sub_end, "text": sub_text.strip()}
                    )
                    st.success(f"추가됨: {sub_text}")
        if st.session_state["subtitles"]:
            st.markdown("**등록된 자막:**")
            to_del = []
            for i, s in enumerate(st.session_state["subtitles"]):
                cols = st.columns([3,1])
                cols[0].markdown(
                    f"`{sec_to_ts(s['start'])}` → `{sec_to_ts(s['end'])}` &nbsp; **{s['text']}**",
                    unsafe_allow_html=True
                )
                if cols[1].button("🗑", key=f"del_sub_{i}"):
                    to_del.append(i)
            for i in reversed(to_del):
                st.session_state["subtitles"].pop(i)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. 배경음
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>4</span> 🎵 배경음 삽입", unsafe_allow_html=True)
    use_bgm = st.toggle("배경음 사용", value=False)
    if use_bgm:
        bgm_file = st.file_uploader("배경음 파일 (MP3, WAV)", type=["mp3","wav"], key="bgm")
        bgm_vol  = st.slider("배경음 볼륨", 0.0, 1.0, 0.3, 0.05)
        if bgm_file:
            tmp_bgm = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(bgm_file.name).suffix)
            tmp_bgm.write(bgm_file.read()); tmp_bgm.flush()
            st.session_state["bgm_path"]   = tmp_bgm.name
            st.session_state["bgm_volume"] = bgm_vol
            st.audio(bgm_file)
        elif "bgm_path" in st.session_state:
            st.session_state["bgm_volume"] = bgm_vol
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. 이미지 삽입
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>5</span> 🖼 이미지 삽입", unsafe_allow_html=True)
    use_img = st.toggle("이미지 삽입 사용", value=False)
    if use_img:
        img_file = st.file_uploader("이미지 (PNG, JPG)", type=["png","jpg","jpeg"], key="img")
        if img_file:
            c1, c2, c3, c4 = st.columns(4)
            img_start = c1.number_input("시작(초)", 0.0, duration, 0.0, 0.5, key="is")
            img_end   = c2.number_input("끝(초)",   0.0, duration, 3.0, 0.5, key="ie")
            img_pos   = c3.selectbox("위치", ["center","top-left","top-right","bottom-left","bottom-right"])
            img_scale = c4.slider("크기 (%)", 10, 100, 30)
            if st.button("➕ 이미지 추가"):
                from PIL import Image
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                pil = Image.open(img_file)
                # Resize
                w, h = pil.size
                new_w = max(50, int(w * img_scale / 100))
                new_h = int(h * new_w / w)
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                pil.save(tmp_img.name)
                st.session_state["images"].append({
                    "start": img_start, "end": img_end,
                    "path": tmp_img.name, "position": img_pos,
                    "opacity": 1.0
                })
                st.success("이미지 추가됨!")
        if st.session_state["images"]:
            st.markdown(f"**등록된 이미지:** {len(st.session_state['images'])}개")
            if st.button("🗑 전체 삭제"):
                st.session_state["images"] = []
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. 인물 트래킹
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### <span class='step-badge'>6</span> 🎯 인물 자동 트래킹", unsafe_allow_html=True)
    use_track = st.toggle("인물 트래킹 사용", value=False)
    if use_track:
        st.info("MediaPipe Pose로 인물의 위치를 추적하여 화면을 자동으로 따라갑니다.")
        track_width = st.slider("화면 크롭 비율 (%)", 40, 100, 60) / 100
        track_smooth = st.slider("움직임 부드러움", 5, 60, 20)
        st.session_state["track_settings"] = {
            "enabled": True,
            "width_ratio": track_width,
            "smooth": track_smooth
        }
    else:
        st.session_state["track_settings"] = {"enabled": False}
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Process & preview ────────────────────────────────────────────────
with col_right:
    st.markdown("### 🚀 편집 처리")

    # Summary
    enabled = []
    cs = st.session_state.get("cut_settings", {})
    ts = st.session_state.get("track_settings", {})
    if cs.get("enabled"): enabled.append("✅ 자동 컷편집")
    if use_sub and st.session_state["subtitles"]:
        enabled.append(f"✅ 자막 {len(st.session_state['subtitles'])}개")
    if use_bgm and "bgm_path" in st.session_state: enabled.append("✅ 배경음")
    if use_img and st.session_state["images"]:
        enabled.append(f"✅ 이미지 {len(st.session_state['images'])}개")
    if ts.get("enabled"): enabled.append("✅ 인물 트래킹")

    if enabled:
        st.markdown("**적용될 편집:**")
        for e in enabled:
            st.markdown(f"- {e}")
    else:
        st.info("왼쪽에서 원하는 편집 기능을 활성화하세요.")

    st.markdown("---")

    if st.button("🎬 편집 시작", type="primary", use_container_width=True):
        if not enabled:
            st.warning("적용할 편집 기능을 하나 이상 선택해주세요.")
        else:
            progress_bar = st.progress(0, text="편집 준비 중...")
            log_area = st.empty()
            current = st.session_state["video_path"]

            try:
                step_total = len(enabled)
                step_done  = 0

                def upd(pct, msg):
                    progress_bar.progress(
                        min(0.99, (step_done / step_total) + pct / step_total),
                        text=msg
                    )

                # STEP A: 자동 컷편집
                if cs.get("enabled"):
                    log_area.markdown("🔍 **무음 구간 분석 중...**")
                    rms_db, chunk_dur = get_audio_rms_per_chunk(current)
                    if rms_db:
                        silents = detect_silent_intervals(
                            rms_db, chunk_dur,
                            cs["db_thresh"], cs["min_silence"]
                        )
                        # Add padding
                        pad = cs.get("padding", 0.1)
                        silents = [
                            (max(0, s+pad), e-pad)
                            for s, e in silents if (e-pad) > (s+pad)
                        ]
                        keeps = silence_to_keep(duration, silents)
                        log_area.markdown(
                            f"✂️ **컷편집:** 무음 {len(silents)}개 구간 제거 "
                            f"({len(keeps)}개 구간 유지)"
                        )
                        if keeps:
                            tmp_out = tempfile.mktemp(suffix=".mp4")
                            cmd = build_concat_filter(keeps, current, tmp_out)
                            ok, err = run_ffmpeg(cmd)
                            if ok:
                                current = tmp_out
                            else:
                                st.warning(f"컷편집 중 오류 (계속 진행): {err[-200:]}")
                    step_done += 1
                    upd(1.0, "컷편집 완료")

                # STEP B: 인물 트래킹
                if ts.get("enabled"):
                    log_area.markdown("🎯 **인물 트래킹 처리 중... (시간이 걸릴 수 있습니다)**")
                    tmp_out = tempfile.mktemp(suffix=".mp4")
                    ok, err = apply_tracking(
                        current, tmp_out,
                        track_width_ratio=ts["width_ratio"],
                        smooth_frames=ts["smooth"],
                        progress_cb=lambda p: upd(p, f"트래킹 처리 중... {int(p*100)}%")
                    )
                    if ok:
                        current = tmp_out
                    else:
                        st.warning(f"트래킹 오류 (계속): {err[-200:]}")
                    step_done += 1
                    upd(1.0, "트래킹 완료")

                # STEP C: 이미지 오버레이
                if use_img and st.session_state["images"]:
                    log_area.markdown("🖼 **이미지 합성 중...**")
                    tmp_out = tempfile.mktemp(suffix=".mp4")
                    ok, err = overlay_images_ffmpeg(
                        current, tmp_out, st.session_state["images"]
                    )
                    if ok:
                        current = tmp_out
                    else:
                        st.warning(f"이미지 합성 오류 (계속): {err[-200:]}")
                    step_done += 1
                    upd(1.0, "이미지 합성 완료")

                # STEP D: 배경음
                if use_bgm and "bgm_path" in st.session_state:
                    log_area.markdown("🎵 **배경음 합성 중...**")
                    tmp_out = tempfile.mktemp(suffix=".mp4")
                    ok, err = add_bgm_ffmpeg(
                        current, tmp_out,
                        st.session_state["bgm_path"],
                        st.session_state.get("bgm_volume", 0.3)
                    )
                    if ok:
                        current = tmp_out
                    else:
                        st.warning(f"배경음 오류 (계속): {err[-200:]}")
                    step_done += 1
                    upd(1.0, "배경음 합성 완료")

                # STEP E: 자막
                if use_sub and st.session_state["subtitles"]:
                    log_area.markdown("💬 **자막 입히는 중...**")
                    tmp_out = tempfile.mktemp(suffix=".mp4")
                    ok, err = add_subtitles_ffmpeg(
                        current, tmp_out, st.session_state["subtitles"]
                    )
                    if ok:
                        current = tmp_out
                    else:
                        st.warning(f"자막 오류 (계속): {err[-200:]}")
                    step_done += 1
                    upd(1.0, "자막 완료")

                # 최종 출력 파일을 /mnt/user-data/outputs 으로 복사
                final_path = "/mnt/user-data/outputs/edited_video.mp4"
                import shutil
                shutil.copy2(current, final_path)
                st.session_state["output_path"] = final_path

                progress_bar.progress(1.0, text="✅ 편집 완료!")
                log_area.empty()

            except Exception as ex:
                st.error(f"처리 중 오류 발생: {ex}")
                import traceback; traceback.print_exc()

    # 결과 미리보기 & 다운로드
    if st.session_state.get("output_path") and os.path.exists(st.session_state["output_path"]):
        st.success("🎉 편집 완료!")
        new_dur = get_video_duration(st.session_state["output_path"])
        st.caption(f"결과 영상 길이: {sec_to_ts(new_dur)}")
        st.video(st.session_state["output_path"])
        with open(st.session_state["output_path"], "rb") as f:
            st.download_button(
                "⬇️ 편집된 영상 다운로드",
                f, "edited_video.mp4", "video/mp4",
                use_container_width=True
            )

    # 초기화
    st.markdown("---")
    if st.button("🔄 전체 초기화", use_container_width=True):
        for k in ["video_path","output_path","subtitles","images","bgm_path","duration"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["subtitles"] = []
        st.session_state["images"]    = []
        st.rerun()
