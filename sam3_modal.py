import modal
import os
import subprocess

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
    .pip_install(
        "matplotlib",
        "pillow",
        "opencv-python",
    )
    .add_local_dir(
        "sam3",
        remote_path="/root/SAM3/sam3",
        copy=True,
    )
    .run_commands(
        "cd /root/SAM3/sam3 && pip install -e '.[notebooks]'",
    )
)

volume = modal.Volume.from_name(
    "sam3-data",
    create_if_missing=True,
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def run_sam3_video(
    video_filename: str,
    prompt_text: str,
):

    import sys
    import pickle
    import cv2
    import numpy as np

    sys.path.insert(
        0,
        "/root/SAM3/sam3",
    )

    from sam3.model.sam3_video_predictor import (
        Sam3VideoPredictor
    )


    # 1. Load SAM3

    predictor = Sam3VideoPredictor()

    video_path = f"/data/{video_filename}"

    # 2. Start SAM3 session


    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": video_path,
    })

    session_id = response["session_id"]

    # 3. Add text prompt

    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt_text,
    })

    # 4. Propagate through video

    outputs_per_frame = {}

    for response in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):

        frame_index = response["frame_index"]

        outputs_per_frame[frame_index] = response["outputs"]

    # 5. Save raw SAM3 results

    with open(
        "/data/results.pkl",
        "wb",
    ) as f:
        pickle.dump(
            outputs_per_frame,
            f,
        )

    # 6. Open original video

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Video: {width}x{height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Frames: {frame_count}"
    )

    # 7. Create results directory

    results_dir = "/data/results"

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    # 8. Render and save every frame

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Get SAM3 output for this frame

        if frame_index in outputs_per_frame:

            outputs = outputs_per_frame[
                frame_index
            ]

            masks = None

            # SAM3 output structure can vary
            if isinstance(outputs, dict):

                if "masks" in outputs:

                    masks = outputs["masks"]

                elif "out_binary_masks" in outputs:

                    masks = outputs[
                        "out_binary_masks"
                    ]

            # Convert mask to numpy

            if masks is not None:

                if hasattr(
                    masks,
                    "detach",
                ):

                    masks = (
                        masks
                        .detach()
                        .cpu()
                        .numpy()
                    )

                masks = np.asarray(
                    masks
                )

                # Remove unnecessary dimensions
                masks = np.squeeze(
                    masks
                )

                # Multiple objects

                if masks.ndim == 3:

                    combined_mask = np.any(
                        masks > 0,
                        axis=0,
                    )

                # Single object

                elif masks.ndim == 2:

                    combined_mask = (
                        masks > 0
                    )

                else:

                    combined_mask = None

                # Render mask

                if combined_mask is not None:

                    combined_mask = (
                        combined_mask
                        .astype(np.uint8)
                    )

                    # Resize if needed
                    if combined_mask.shape != (
                        height,
                        width,
                    ):

                        combined_mask = cv2.resize(
                            combined_mask,
                            (
                                width,
                                height,
                            ),
                            interpolation=(
                                cv2.INTER_NEAREST
                            ),
                        )

                    # Green segmentation overlay
                    overlay = frame.copy()

                    overlay[
                        combined_mask > 0
                    ] = (
                        0,
                        255,
                        0,
                    )

                    frame = cv2.addWeighted(
                        frame,
                        0.7,
                        overlay,
                        0.3,
                        0,
                    )

                    # Draw segmentation contour

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

        # Save JPG

        output_filename = (
            f"frame_{frame_index + 1}.jpg"
        )

        output_path = os.path.join(
            results_dir,
            output_filename,
        )

        cv2.imwrite(
            output_path,
            frame,
        )

        print(
            f"Saved: {output_filename}"
        )

        frame_index += 1

    # 9. Close video

    cap.release()
    # ---------------------------------------------------------
    # 10. Commit Modal Volume

    volume.commit()

    return (
        f"Processed {frame_index} frames. "
        f"Results saved to /data/results/"
    )


@app.local_entrypoint()
def main():

    # LOCAL PATH

    project_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    video_path = os.path.join(
        project_dir,
        "test",
        "sample_banana360.mp4",
    )

    # Check that video exists locally

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    print(
        f"Using local video:\n{video_path}"
    )

    # Upload video to Modal Volume

    print(
        "Uploading sample_banana360.mp4 "
        "to Modal..."
    )

    subprocess.run(
        [
            "modal",
            "volume",
            "put",
            "sam3-data",
            video_path,
            "sample_banana360.mp4",
        ],
        check=True,
    )

    print(
        "Video uploaded successfully."
    )

    # Run SAM3 remotely

    result = run_sam3_video.remote(
        video_filename="sample_banana360.mp4",
        prompt_text="banana",
    )

    print(result)

    # LOCAL RESULTS DIRECTORY

    local_results = os.path.join(
        project_dir,
        "results",
    )

    os.makedirs(
        local_results,
        exist_ok=True,
    )

    # Download generated frames

    print(
        "Downloading frames..."
    )

    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "sam3-data",
            "results",
            local_results,
        ],
        check=True,
    )

    print()

    print(
        "DONE"
    )

    print(
        f"Frames saved to:\n{local_results}"
    )