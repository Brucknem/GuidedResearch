//
// Created by brucknem on 19.01.21.
//

#include "KeyframeHistory.hpp"

namespace providentia {
    namespace calibration {
#pragma region KeyframeHistory

        void KeyframeHistory::addKeyframe(const cv::cuda::GpuMat &frame) {
            keyframes.emplace_back(new providentia::features::SURFFeatureDetector(100));
            keyframes.back()->detect(frame);
            if (!shouldAddKeyframe()) {
                keyframes.pop_back();
            }
        }

        const cv::Mat &KeyframeHistory::draw() {
            if (keyframes.empty()) {
                drawFrame = cv::Mat::ones(cv::Size(1, 1), CV_32F);
                return drawFrame;
            }
            drawFrame = keyframes[0]->draw();

            for (auto it = keyframes.begin() + 1; it != keyframes.end(); ++it) {
                cv::vconcat(std::vector<cv::Mat>{drawFrame, it->get()->draw()}, drawFrame);
            }

            return drawFrame;
        }

        const std::vector<std::shared_ptr<providentia::features::FeatureDetectorBase>> &
        KeyframeHistory::getKeyframes() const {
            return keyframes;
        }

#pragma endregion KeyframeHistory

#pragma region NumberBasedKeyframeHistory

        NumberBasedKeyframeHistory::NumberBasedKeyframeHistory(int numberOfFrames)
                : sequenceDelta(
                numberOfFrames) {}

        bool NumberBasedKeyframeHistory::shouldAddKeyframe() {
            if (currentSequenceDelta++ % sequenceDelta == 0) {
                currentSequenceDelta = 1;
                return true;
            }
            return false;
        }

#pragma endregion NumberBasedKeyframeHistory
    }
}

