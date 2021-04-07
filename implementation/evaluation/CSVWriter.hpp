#ifndef CAMERASTABILIZATION_CSVWRITER_HPP
#define CAMERASTABILIZATION_CSVWRITER_HPP

#include <iostream>
#include <fstream>
#include <utility>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#include "ObjectTracking.hpp"
/**
 * https://stackoverflow.com/questions/25201131/writing-csv-files-from-c
 */

namespace providentia {
	namespace evaluation {
		class TrackerWrapper;

		class CSVWriter {
			std::ofstream fs_;
			const std::string separator_;
		public:
			explicit CSVWriter() = default;

			explicit CSVWriter(const std::string &filename, const std::string &separator = ",");

			explicit CSVWriter(const std::string &filename, bool append, std::string separator = ",");

			explicit CSVWriter(const boost::filesystem::path &filename, const std::string &separator = ",");

			explicit CSVWriter(const boost::filesystem::path &filename, bool append, std::string separator = ",");

			~CSVWriter();

			void flush();

			void newline();

			CSVWriter &operator<<(CSVWriter &(*val)(CSVWriter &));

			CSVWriter &operator<<(const char *val);

			CSVWriter &operator<<(const std::string &val);

			CSVWriter &operator<<(const cv::Rect &val);

			CSVWriter &operator<<(const cv::Point2d &val);

			CSVWriter &operator<<(const Eigen::Vector3d &val);

			CSVWriter &operator<<(const providentia::evaluation::TrackerWrapper &val);

			template<typename T>
			CSVWriter &operator<<(const T &val);

		};

		CSVWriter &newline(CSVWriter &file);

		CSVWriter &flush(CSVWriter &file);

		CSVWriter &rect(CSVWriter &file, std::string name);

		CSVWriter &point(CSVWriter &file, std::string name);

	}
}
#endif // CAMERASTABILIZATION_CSVWRITER_HPP
