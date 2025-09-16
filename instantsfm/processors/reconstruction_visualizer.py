import time
import numpy as np
import viser
import viser.transforms as tf
import threading
from scipy.spatial.transform import Rotation
from typing import Optional, List, Tuple, Dict, Any

class ReconstructionVisualizer:
    """Visualizer for Structure from Motion reconstruction process.
    
    This visualizer displays the evolving point cloud and camera poses during
    the SfM process, with smooth transitions between updates.
    """
    
    def __init__(
        self,
        min_update_interval: float = 0.1,  # Minimum time between visual updates in seconds
        point_size: float = 0.05,  # Increased default point size
        show_axes: bool = True,
        frustum_scale: float = 0.2,  # Increased default frustum scale
    ):
        """Initialize the SfM visualization.
        
        Args:
            min_update_interval: Minimum time between visual updates (seconds)
            point_size: Size of the 3D points
            show_axes: Whether to show coordinate axes
            frustum_scale: Scale of camera frustums
        """
        self.server = viser.ViserServer()
        self.min_update_interval = min_update_interval
        self.point_size = point_size
        self.show_axes = show_axes
        self.frustum_scale = frustum_scale
        
        # Initialize visualization elements
        self.points_handle: Optional[viser.PointCloudHandle] = None
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        
        # For smooth updates
        self.last_update_time = 0.0
        self.pending_update = False
        self.update_lock = threading.Lock()
        
        # Data storage
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.cameras: Dict[int, Dict[str, Any]] = {}
        
        # Configure visualization
        self._setup_visualization()
        
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _setup_visualization(self):
        """Set up the initial visualization elements."""
        # Add world axes
        self.server.scene.world_axes.visible = self.show_axes
        
        # Add GUI elements
        with self.server.gui.add_folder("Visualization Controls"):
            self.gui_point_size = self.server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=self.point_size,
            )
            
            self.gui_show_cameras = self.server.gui.add_checkbox(
                "Show Cameras", 
                initial_value=True
            )
            
            self.gui_frustum_scale = self.server.gui.add_slider(
                "Frustum Scale",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=self.frustum_scale,
            )
        
        # Connect GUI callbacks
        @self.gui_point_size.on_update
        def _(_):
            self.point_size = self.gui_point_size.value
            if self.points_handle is not None:
                self.points_handle.point_size = self.point_size
        
        @self.gui_show_cameras.on_update
        def _(_):
            for camera_id, handle in self.camera_handles.items():
                handle.visible = self.gui_show_cameras.value
        
        @self.gui_frustum_scale.on_update
        def _(_):
            self.frustum_scale = self.gui_frustum_scale.value
            for camera_id, handle in self.camera_handles.items():
                handle.scale = self.frustum_scale
    
    def add_step(
        self,
        cameras,
        images,
        tracks,
        step_name: str = None,
    ):
        """Add results from a new SfM computation step
        
        Args:
            cameras: List of camera intrinsic objects
            images: List of image objects (containing pose information)
            tracks: List/dictionary of 3D point objects
            step_name: Name of current step (optional)
        """
        # Process point cloud data
        points_list = []
        colors_list = []
        
        # Extract points and colors from tracks
        if isinstance(tracks, dict):
            # If tracks is a dictionary (key-value pairs)
            for track in tracks.values():
                if hasattr(track, 'xyz') and track.xyz is not None:
                    points_list.append(track.xyz)
                    if hasattr(track, 'color') and np.any(track.color):
                        colors_list.append(track.color)
                    else:
                        colors_list.append(np.array([255, 0, 0], dtype=np.uint8))
        else:
            # If tracks is a list
            for track in tracks:
                if hasattr(track, 'xyz') and track.xyz is not None:
                    points_list.append(track.xyz)
                    if hasattr(track, 'color') and np.any(track.color):
                        colors_list.append(track.color)
                    else:
                        colors_list.append(np.array([255, 0, 0], dtype=np.uint8))
        
        # Process camera data
        cameras_dict = {}
        
        for i, img in enumerate(images):
            # Skip unregistered cameras
            if not hasattr(img, 'is_registered') or not img.is_registered:
                continue
                
            # Get camera parameters
            if not hasattr(img, 'world2cam') or img.world2cam is None:
                continue
                
            R = img.world2cam[:3, :3]
            R = R.T
            t = img.center()
            
            # Ensure camera ID exists
            if not hasattr(img, 'cam_id') or img.cam_id is None or img.cam_id >= len(cameras):
                continue
                
            cam = cameras[img.cam_id]
            
            # Calculate FOV
            fov = 2 * np.arctan2(cam.height / 2, cam.focal_length[1]) if hasattr(cam, 'focal_length') else 0.8
            aspect = cam.width / cam.height if hasattr(cam, 'width') and hasattr(cam, 'height') else 1.0
            
            # Create quaternion from rotation matrix
            xyzw = Rotation.from_matrix(R).as_quat()
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # Convert to wxyz format
            
            # Create camera dictionary
            cameras_dict[i] = {
                'position': t,
                'wxyz': wxyz,
                'fov': fov,
                'aspect': aspect,
            }
            
            # Add image data if available
            if hasattr(img, 'image_data') and img.image_data is not None:
                cameras_dict[i]['image'] = img.image_data
        
        # Convert to numpy arrays
        if points_list and colors_list:
            points_array = np.array(points_list, dtype=np.float32)
            colors_array = np.array(colors_list, dtype=np.uint8)
            
            # Update visualization through existing add_step method
            with self.update_lock:
                self.points = points_array
                self.colors = colors_array
                self.cameras = cameras_dict
                self.pending_update = True
                
            return True
        else:
            print("No valid points or cameras to visualize")
            return False

    def _update_visualization(self):
        """Update the visualization with the latest data."""
        if self.points is not None and self.colors is not None:
            # Convert to numpy arrays if they aren't already
            points_array = np.asarray(self.points, dtype=np.float32)
            colors_array = np.asarray(self.colors, dtype=np.uint8)
            
            # Update or create point cloud
            if self.points_handle is None:
                self.points_handle = self.server.scene.add_point_cloud(
                    "/reconstruction/points",
                    points=points_array,
                    colors=colors_array,
                    point_size=self.point_size,
                    point_shape="circle",
                )
            else:
                self.points_handle.points = points_array
                self.points_handle.colors = colors_array
                self.points_handle.point_size = self.point_size
        
        # Process cameras
        current_camera_ids = set(self.cameras.keys())
        existing_camera_ids = set(self.camera_handles.keys())
        
        # Remove cameras that are no longer present
        for camera_id in existing_camera_ids - current_camera_ids:
            self.camera_handles[camera_id].remove()
            del self.camera_handles[camera_id]
        
        # Update or add cameras
        for camera_id in current_camera_ids:
            camera = self.cameras[camera_id]
            
            if camera_id in self.camera_handles:
                # Update existing camera
                handle = self.camera_handles[camera_id]
                with self.server.atomic():
                    handle.wxyz = camera['wxyz']
                    handle.position = camera['position']
                    handle.scale = self.frustum_scale
                    if 'image' in camera:
                        handle.image = camera['image']
            else:
                frustum = self.server.scene.add_camera_frustum(
                    f"/reconstruction/camera_{camera_id}",
                    fov=camera.get('fov', 0.8),
                    aspect=camera.get('aspect', 1.0),
                    scale=self.frustum_scale,
                    wxyz=camera['wxyz'],
                    position=camera['position'],
                    image=camera.get('image', None),
                    visible=self.gui_show_cameras.value,
                )
                self.camera_handles[camera_id] = frustum
        
        # Flush changes to ensure they're sent to clients
        self.server.flush()
    
    def _update_loop(self):
        """Background thread that handles smooth visual updates."""
        while self.running:
            current_time = time.time()
            
            with self.update_lock:
                should_update = (
                    self.pending_update and 
                    (current_time - self.last_update_time) >= self.min_update_interval
                )
                
                if should_update:
                    self.pending_update = False
                    self.last_update_time = current_time
            
            if should_update:
                self._update_visualization()
            
            # Sleep a small amount to prevent CPU hogging
            time.sleep(0.01)
    
    def close(self):
        """Clean up resources used by the visualizer."""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        # The server will continue running until the program exits
    
    def run_server(self, blocking=False):
        """Start the visualization server.
        
        Args:
            blocking: If True, this function will block until the server
                     is terminated. If False, the server will run in the
                     background.
        """
        # The server is actually already running from __init__,
        # we just need this method to match the expected interface
        
        if blocking:
            # If blocking, we just sleep indefinitely or until interrupted
            try:
                while self.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.close()