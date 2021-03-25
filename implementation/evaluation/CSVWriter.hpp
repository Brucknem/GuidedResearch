#ifndef CAMERASTABILIZATION_CSVWRITER_HPP
#define CAMERASTABILIZATION_CSVWRITER_HPP

#include <iostream>
#include <fstream>
#include <utility>

/**
 * https://stackoverflow.com/questions/25201131/writing-csv-files-from-c
 */

namespace providentia {
	namespace evaluation {

		class CSVWriter {
			std::ofstream fs_;
			const std::string separator_;
		public:
			explicit CSVWriter() = default;

			explicit CSVWriter(const std::string &filename, const std::string &separator = ",");

			explicit CSVWriter(const std::string &filename, bool append, std::string separator = ",");

			~CSVWriter();

			void flush();

			void newline();

			CSVWriter &operator<<(CSVWriter &(*val)(CSVWriter &));

			CSVWriter &operator<<(const char *val);

			CSVWriter &operator<<(const std::string &val);

			template<typename T>
			CSVWriter &operator<<(const T &val);

		};

		CSVWriter &newline(CSVWriter &file);

		CSVWriter &flush(CSVWriter &file);
	}
}
#endif // CAMERASTABILIZATION_CSVWRITER_HPP
