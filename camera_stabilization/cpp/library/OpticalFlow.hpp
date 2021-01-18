//
// Created by brucknem on 18.01.21.
//

#ifndef CAMERASTABILIZATION_OPTICALFLOW_HPP
#define CAMERASTABILIZATION_OPTICALFLOW_HPP

#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaoptflow.hpp"

#include "TimeMeasurable.hpp"


namespace providentia {
    namespace opticalflow {

        /**
         * Base class for dense optical flow algorithms
         */
        class DenseOpticalFlow : public providentia::utils::TimeMeasurable {
        protected:
            /**
             * The current and previous GPU frame.
             */
            cv::cuda::GpuMat currentFrame, previousFrame;

            /**
             * The calculated dense optical flow.
             */
            cv::Mat denseOpticalFlowCPU;
            cv::cuda::GpuMat denseOpticalFlowGPU;

            /**
             * Buffer matrices.
             */
            cv::Mat hsv, magnitude, angle, angles;

            /**
             * The three channels of the optical flow.
             */
            std::vector<cv::Mat> flowParts{3};

            /**
             * The CUDA stream for async calculations on GPU.
             */
            cv::cuda::Stream stream;

            /**
             * Color frames for visualization.
             */
            cv::Mat _hsv[3], hsv8, bgr;

            /**
             * @constructor
             */
            explicit DenseOpticalFlow() : providentia::utils::TimeMeasurable("Dense Optical Flow", 1) {}

            /**
             * Initializes the frame buffers on the first input frame.
             */
            void initialize();

            /**
             * Algorithm specific calculation function.
             */
            virtual void specificCalculate() = 0;

        public:

            /**
             * @destructor
             */
            virtual ~DenseOpticalFlow();

            /**
             * @get the mean of the magnitudes over the flow field.
             */
            double getMagnitudeMean();


            /**
             * @get the mean of the angles over the flow field.
             */
            double getAngleMean();

            /**
             * Draws the optical flow field as BGR image.
             * @return
             */
            const cv::Mat &draw() const;

            /**
             * Main calculation function.
             *
             * @param _frame The frame to process.
             */
            void calculate(const cv::cuda::GpuMat &_frame);
        };

        /**
         * Wrapper of the Farneback dense optical flow algorithm.
         */
        class FarnebackDenseOpticalFlow : public DenseOpticalFlow {
        private:

            /**
             * The algorithm implementation.
             */
            cv::Ptr<cv::cuda::FarnebackOpticalFlow> opticalFlow;

        public:
            /**
             * @constructor
             */
            explicit FarnebackDenseOpticalFlow();

            /**
             * @destructor
             */
            ~FarnebackDenseOpticalFlow() override;

            void specificCalculate() override;
        };

    }
}


#endif //CAMERASTABILIZATION_OPTICALFLOW_HPP
