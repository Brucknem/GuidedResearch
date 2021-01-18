//
// Created by brucknem on 18.01.21.
//

#ifndef CAMERASTABILIZATION_CSVWRITER_HPP
#define CAMERASTABILIZATION_CSVWRITER_HPP

#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <utility>
#include <map>
#include <chrono>
#include <cstdarg>
#include <fstream>

namespace providentia {
    namespace utils {

        /**
         * A CSV writer to ease evaluations.
         */
        class CSVWriter {
        private:

            /**
             * The CSV file stream.
             */
            std::ofstream fileStream;

            /**
             * The current line to write to the CSV file.
             */
            std::stringstream currentLine;

            /**
             * Flag to switch writing on and off situational.
             */
            bool write;

        public:

            /**
             * @constructor Opens the CSV file stream.
             *
             * @param path The absolute path of the CSV file.
             * @param _write Flag whether to write or not.
             */
            explicit CSVWriter(const std::string &path, bool write = true);

            /**
             * @desctructor Writes and closes the file.
             */
            virtual ~CSVWriter();

            /**
             * Write the last name and start a new one.
             */
            void newLine();

            /**
             * Generic append function for a variable amount of data.
             */
            template<typename Arg, typename... Args>
            void append(Arg &&arg, Args &&... args);
        };
    }
}


#endif //CAMERASTABILIZATION_CSVWRITER_HPP
