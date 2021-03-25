//
// Created by brucknem on 18.01.21.
//

#include "CSVWriter.hpp"

#include <utility>

namespace providentia {
	namespace evaluation {

		template<typename T>
		CSVWriter &CSVWriter::operator<<(const T &val) {
			fs_ << val << separator_;
			return *this;
		}

		CSVWriter::CSVWriter(const std::string &filename, bool append, std::string separator)
			: fs_(), separator_(std::move(separator)) {
			fs_.exceptions(std::ios::failbit | std::ios::badbit);
			if (append) {
				fs_.open(filename, std::ofstream::app);
			} else {
				fs_.open(filename);
			}
		}

		CSVWriter::CSVWriter(const std::string &filename, const std::string &separator) :
			CSVWriter(filename, false, separator) {}

		CSVWriter::~CSVWriter() {
			flush();
			fs_.close();
		}

		void CSVWriter::flush() {
			fs_.flush();
		}

		void CSVWriter::newline() {
			fs_ << std::endl;
		}

		CSVWriter &CSVWriter::operator<<(CSVWriter &(*val)(CSVWriter &)) {
			return val(*this);
		}

		CSVWriter &CSVWriter::operator<<(const char *val) {
			fs_ << '"' << val << '"' << separator_;
			return *this;
		}

		CSVWriter &CSVWriter::operator<<(const std::string &val) {
			fs_ << '"' << val << '"' << separator_;
			return *this;
		}

		template CSVWriter &CSVWriter::operator<<(const int &);

		template CSVWriter &CSVWriter::operator<<(const double &);

		template CSVWriter &CSVWriter::operator<<(const float &);

		CSVWriter &newline(CSVWriter &file) {
			file.newline();
			return file;
		}

		CSVWriter &flush(CSVWriter &file) {
			file.flush();
			return file;
		}
	}
}
