//
// Created by brucknem on 21.12.20.
//

#ifndef DYNAMICSTABILIZATION_UTILS_HPP
#define DYNAMICSTABILIZATION_UTILS_HPP

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
         * Class to record timestamps and measure durations of algorithm steps.
         */
        class TimeMeasurable {
        private:
            std::string name;
            std::vector<std::pair<std::string, std::chrono::milliseconds>> timestamps;
            std::vector<std::pair<std::string, long>> durations;
            std::chrono::milliseconds previous{};
            int verbosity;


        public:
            /**
             * Constructor
             *
             * @param name The print name of the instance.
             */
            explicit TimeMeasurable(std::string name = "Unnamed", int verbosity = 1) : name(std::move(name)),
                                                                                       verbosity(verbosity) {
                clear();
            }

            void setName(std::string _name) {
                name = std::move(_name);
            }

            void setVerbosity(int _verbosity) {
                verbosity = _verbosity;
            }

            void setNameAndVerbosity(std::string _name, int _verbosity) {
                setName(std::move(_name));
                setVerbosity(verbosity);
            }

            /**
             * Gets the current unix timestamp in milliseconds since 01.01.1970.

             * @return The total milliseconds since 01.01.1970.
             */
            static std::chrono::milliseconds now() {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch());
            }

            /**
             * Clears the buffers and adds a start timestamp.
             */
            void clear() {
                timestamps.clear();
                durations.clear();
                previous = now();
                addTimestamp("Start");
            }

            /**
             * Adds a timestamp.
             *
             * @param measurementName The name of the measurement.
             */
            void addTimestamp(const std::string &measurementName = "Unnamed", int minVerbosity = 0) {
                if (verbosity < minVerbosity) {
                    return;
                }
                auto now = providentia::utils::TimeMeasurable::now();
                timestamps.emplace_back(measurementName, now);
                durations.emplace_back(std::make_pair(measurementName, (now - previous).count()));
                previous = providentia::utils::TimeMeasurable::now();
            }

            /**
             * Gets the total milliseconds duration from the last clear to the latest added timestamp.

             * @return The total milliseconds.
             */
            long getTotalMilliseconds() {
                return (timestamps[timestamps.size() - 1].second - timestamps[0].second).count();
            }

            /**
             * Converts the durations between the steps into a readable format.
             *
             * @return The formatted durations.
             */
            std::string durations_str() {
                std::stringstream ss;

                for (const auto &duration : durations) {
                    ss << "[" << name << "] " << duration.second << "ms : " << duration.first << std::endl;
                }

                return ss.str();
            }
        };

        class CsvWriter {
        private:
            std::ofstream fileStream;
            std::stringstream currentLine;
            bool write;

        public:
            explicit CsvWriter(const std::string &path, bool _write = true) : write(_write) {
                if (!write) {
                    return;
                }
                fileStream.open(path);
                fileStream.close();
                fileStream.open(path, std::ios_base::app); // append instead of overwrite
            }

            ~CsvWriter() {
                if (!write) {
                    return;
                }
                fileStream.close();
            }

            void newLine() {
                if (!write) {
                    return;
                }
                std::string line = currentLine.str();
                currentLine.str("");
                line.pop_back();
                fileStream << line << std::endl;
            }

            template<typename Arg, typename... Args>
            void append(Arg &&arg, Args &&... args) {
                if (!write) {
                    return;
                }
                currentLine << std::forward<Arg>(arg);
                using expander = int[];
                (void) expander{0, (void(currentLine << ',' << std::forward<Args>(args)), 0)...};
                currentLine << ",";
            }
        };
    }
}
#endif //DYNAMICSTABILIZATION_UTILS_HPP
