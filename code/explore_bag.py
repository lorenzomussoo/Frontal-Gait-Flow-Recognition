import pyrealsense2 as rs
import numpy as np
import cv2
import os
from rich.console import Console
from rich.panel import Panel

console = Console()


BAG_FILE = "/Volumes/LaCie/GAIT/FirstRun/RGBD/Alessio/StairsDown/20240614_180533.bag"

def analyze_bag_file(bag_path):
    if not os.path.exists(bag_path):
        console.print(Panel.fit(f"[bold red]ERROR: File not found![/bold red]\n{bag_path}"))
        return

    console.print(Panel.fit(f"[bold gradient(cyan,magenta)]--- Analyzing ROSBAG RealSense ---[/bold gradient(cyan,magenta)]\nFile: {os.path.basename(bag_path)}"))

    pipeline = rs.pipeline()
    config = rs.config()
    
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)

    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False) 

        console.print("[green]File opened successfully. Analyzing streams...[/green]")
        
        streams = profile.get_streams()
        for stream in streams:
            v_profile = stream.as_video_stream_profile()
            if v_profile:
                console.print(f"  -> Stream found: [bold]{stream.stream_name()}[/bold] | "
                              f"Res: {v_profile.width()}x{v_profile.height()} | "
                              f"FPS: {v_profile.fps()} | "
                              f"Format: {stream.format()}")

        colorizer = rs.colorizer()

        success, frames = pipeline.try_wait_for_frames(timeout_ms=5000)
        if not success:
            console.print("[red]Unable to read initial frames.[/red]")
            return
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            console.print("[yellow]Warning: Depth or Color frame missing in the first packet.[/yellow]")
        else:
            depth_colorized = colorizer.process(depth_frame)
            depth_image_viz = np.asanyarray(depth_colorized.get_data())

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            console.print("\n[bold]Opening a window with frame preview... (Press any key to close)[/bold]")
            
            if depth_image_viz.shape != color_image.shape:
                d_h, d_w = depth_image_viz.shape[:2]
                color_image = cv2.resize(color_image, (d_w, d_h))

            comparison = np.hstack((color_image, depth_image_viz))
            
            cv2.imshow(f"Preview: {os.path.basename(bag_path)} (RGB vs Depth)", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            console.print("[cyan]Window closed.[/cyan]")

            depth_data_raw = np.asanyarray(depth_frame.get_data())
            dist_center = depth_frame.get_distance(depth_frame.get_width() // 2, depth_frame.get_height() // 2)
            console.print(Panel.fit(f"""[bold]Depth Frame Statistics:[/bold]
                Max Value (Raw): {np.max(depth_data_raw)}
                Min Value (Raw): {np.min(depth_data_raw)}
                Distance to center (m): {dist_center:.2f} m
                Data type: {depth_data_raw.dtype} (Crucial: 16-bit, must be converted for OpenCV standard)"""))

    except RuntimeError as e:
        console.print(f"[bold red]Runtime RealSense Error:[/bold red] {e}")
        console.print("[yellow]Verify that the file path is correct and that the file is not corrupted.[/yellow]")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    analyze_bag_file(BAG_FILE)