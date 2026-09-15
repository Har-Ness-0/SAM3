import modal

app = modal.App("sam3-video")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "git-lfs",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /root/sam3",
        "cd /root/sam3 && pip install -e '.[notebooks]'",
    )
    .pip_install(
        "matplotlib",
        "pillow",
        "opencv-python",
    )
)

volume = modal.Volume.from_name(
    "sam3-data",
    create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def run_sam3_video(video_filename: str, prompt_text: str):

    import sys
    import pickle
    import cv2
    import numpy as np

    sys.path.insert(0, "/root/sam3")

    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    # ---------------------------------------------------------
    # 1. Load SAM3
    # ---------------------------------------------------------

    predictor = Sam3VideoPredictor()

    video_path = f"/data/{video_filename}"

    # ---------------------------------------------------------
    # 2. Start SAM3 session
    # ---------------------------------------------------------

    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": video_path,
    })

    session_id = response["session_id"]

    # ---------------------------------------------------------
    # 3. Add text prompt
    # ---------------------------------------------------------

    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt_text,
    })

    # ---------------------------------------------------------
    # 4. Propagate through video
    # ---------------------------------------------------------

    outputs_per_frame = {}

    for response in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):

        frame_index = response["frame_index"]

        outputs_per_frame[frame_index] = response["outputs"]

    # ---------------------------------------------------------
    # 5. Save raw SAM3 results
    # ---------------------------------------------------------

    with open("/data/results.pkl", "wb") as f:
        pickle.dump(outputs_per_frame, f)

    # ---------------------------------------------------------
    # 6. Open original video
    # ---------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames: {frame_count}")

    # ---------------------------------------------------------
    # 7. Create output MP4
    # ---------------------------------------------------------

    output_path = "/data/result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    # ---------------------------------------------------------
    # 8. Render segmentation masks
    # ---------------------------------------------------------

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index in outputs_per_frame:

            outputs = outputs_per_frame[frame_index]

            # SAM3 output structure can vary depending on version.
            # Try to locate masks.
            masks = None

            if isinstance(outputs, dict):

                if "masks" in outputs:
                    masks = outputs["masks"]

                elif "out_binary_masks" in outputs:
                    masks = outputs["out_binary_masks"]

            if masks is not None:

                if hasattr(masks, "detach"):
                    masks = masks.detach().cpu().numpy()

                masks = np.asarray(masks)

                # Remove unnecessary dimensions
                masks = np.squeeze(masks)

                # If multiple objects exist
                if masks.ndim == 3:

                    combined_mask = np.any(
                        masks > 0,
                        axis=0,
                    )

                elif masks.ndim == 2:

                    combined_mask = masks > 0

                else:

                    combined_mask = None

                if combined_mask is not None:

                    combined_mask = combined_mask.astype(np.uint8)

                    # Resize if necessary
                    if combined_mask.shape != (height, width):

                        combined_mask = cv2.resize(
                            combined_mask,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # Create overlay
                    overlay = frame.copy()

                    # Green segmentation region
                    overlay[combined_mask > 0] = (
                        0,
                        255,
                        0,
                    )

                    # Blend with original
                    frame = cv2.addWeighted(
                        frame,
                        0.7,
                        overlay,
                        0.3,
                        0,
                    )

                    # Draw contour
                    contours, _ = cv2.findContours(
                        combined_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    cv2.drawContours(
                        frame,
                        contours,
                        -1,
                        (0, 255, 0),
                        2,
                    )

        writer.write(frame)

        frame_index += 1

    # ---------------------------------------------------------
    # 9. Close video
    # ---------------------------------------------------------

    cap.release()
    writer.release()

    return (
        f"Processed {len(outputs_per_frame)} frames. "
        f"Saved /data/result.mp4 and /data/results.pkl"
    )


@app.local_entrypoint()
def main():

    result = run_sam3_video.remote(
        video_filename="bedroom.mp4",
        prompt_text="person",
    )

    print(result)