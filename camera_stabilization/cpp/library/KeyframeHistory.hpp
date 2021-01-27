//
// Created by brucknem on 19.01.21.
//

#ifndef CAMERASTABILIZATION_KEYFRAMEHISTORY_HPP
#define CAMERASTABILIZATION_KEYFRAMEHISTORY_HPP

#include <vector>
#include "opencv2/opencv.hpp"
#include "FeatureDetection.hpp"
#include "boost/shared_ptr.hpp"

namespace providentia {
    namespace calibration {
        class KeyframeHistory {
        protected:
            std::vector<std::shared_ptr<providentia::features::FeatureDetectorBase>> keyframes;
        public:
            const std::vector<std::shared_ptr<providentia::features::FeatureDetectorBase>> &getKeyframes() const;

        protected:

            cv::Mat drawFrame;

            virtual bool shouldAddKeyframe() = 0;

        public:
            void addKeyframe(const cv::cuda::GpuMat &frame);

            virtual ~KeyframeHistory() = default;

            const cv::Mat &draw();

        };

        class NumberBasedKeyframeHistory : public KeyframeHistory {
        private:
            int sequenceDelta;
            int currentSequenceDelta = 0;

            bool shouldAddKeyframe() override;

        public:
            explicit NumberBasedKeyframeHistory(int numberOfFrames);
        };
    }
}

#endif //CAMERASTABILIZATION_KEYFRAMEHISTORY_HPP
