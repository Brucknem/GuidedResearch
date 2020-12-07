from src.teknomo_fernandez import *
from src.filesystem_image_provider import ImageBasedVideoCapture
from src.rendering import Renderer
from src.background_segmentation import *

if __name__ == '__main__':
    np.random.seed(0)
    cap = ImageBasedVideoCapture('/mnt/nextcloud/tum/Master/5. Semester/Guided Research/videos/s40_n_cam_far/stamp',
                                 loop=False)
    cuda_stream = cv.cuda_Stream()

    renderer = Renderer()

    filters = [
        # CudaOpenFilter(),
        # CudaErodeFilter(),
        # CudaOpenFilter(),
        # CudaDilateFilter()
    ]

    algorithms = [
        MOGAlgorithm(history=1000),
        MOG2Algorithm(history=1000, varThreshold=0, detectShadows=False)
    ]

    scale_factor = 2

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = Frame(frame)
        size = frame.size() / scale_factor
        frame.resize(size)

        results = {}
        for algorithm in algorithms:
            result = Frame(algorithm.apply(frame))

            for filter in filters:
                result = filter.apply(result)
            result.add_text(algorithm.name, 'orange')
            results[algorithm.name] = result

        if len(results) == 0:
            result = frame
            if not renderer.render(result):
                break
            continue

        masks = []
        processed_images = [frame] * 2
        for name, result in results.items():
            # processed_images.append((~result * (1. / 255.)) * frame)
            masks.append(result)

        frames = []
        positions = []
        for index, frame_masks in enumerate(zip(processed_images, masks)):
            if index == 0:
                frames = [frame_masks[0], frame_masks[1]]
                positions = [(0, 0), (frames[0].size()[0], 0)]
                continue

            frames.append(frame_masks[0])
            frames.append(frame_masks[1])
            previous_index = index * 2 - 2
            previous_size = frames[previous_index].size()
            positions.append(positions[previous_index] + np.array([0,previous_size[1]]))
            positions.append(positions[previous_index] + frames[previous_index].size())

        if not renderer.render(frames, positions):
            break

        previous_frame = frame
        i += 1
