import time
from argparse import ArgumentParser

from instantsfm.controllers.config import Config
from instantsfm.controllers.data_reader import ReadData, ReadColmapDatabase, ReadDepths
from instantsfm.controllers.global_mapper import SolveGlobalMapper
from instantsfm.controllers.reconstruction_writer import WriteGlomapReconstruction

def run_sfm():
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the data folder')
    parser.add_argument('--enable_gui', action='store_true', help='Enable GUI for visualization')
    parser.add_argument('--disable_depths', action='store_true', help='Disable the use of depths if available')
    parser.add_argument('--manual_config_name', help='Name of the manual configuration file')
    mapper_args = parser.parse_args()

    path_info = ReadData(mapper_args.data)
    if not path_info:
        print('Invalid data path, please check the provided path')
        return
    
    view_graph, cameras, images, feature_name = ReadColmapDatabase(path_info.database_path)
    if view_graph is None or cameras is None or images is None:
        return
    if path_info.depth_path and not mapper_args.disable_depths:
        depths = ReadDepths(path_info.depth_path)
    else:
        depths = None

    # enable different configs for different feature handlers and image numbers
    start_time = time.time()
    config = Config(feature_name, mapper_args.manual_config_name, len(images))
    config.ENABLE_GUI = mapper_args.enable_gui

    cameras, images, tracks = SolveGlobalMapper(view_graph, cameras, images, config, depths=depths)
    print('Reconstruction done in', time.time() - start_time, 'seconds')
    WriteGlomapReconstruction(path_info.output_path, cameras, images, tracks, path_info.image_path)
    print('Reconstruction written to', path_info.output_path)

    if config.ENABLE_GUI:
        # block until the GUI is closed
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Visualization server terminated by user.")

def entrypoint():
    # Entry point for pyproject.toml
    run_sfm()
    
if __name__ == '__main__':
    entrypoint()