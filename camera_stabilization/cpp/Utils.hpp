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

namespace providentia {
    namespace utils {
        class TimeMeasurable {
        private:
            std::string name;
            std::vector<std::pair<std::string, std::chrono::milliseconds>> timestamps;
            std::vector<std::pair<std::string, long>> durations;
            std::chrono::milliseconds previous{};

        public:
            explicit TimeMeasurable(std::string name = "Unnamed") : name(std::move(name)) {
                clear();
            }

            static std::chrono::milliseconds now() {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch());
            }

            void clear() {
                timestamps.clear();
                durations.clear();
                previous = now();
                addTimestamp("Start");
            }

            void addTimestamp(const std::string &measurementName = "Unnamed") {
                auto now = providentia::utils::TimeMeasurable::now();
                timestamps.emplace_back(measurementName, now);
                durations.emplace_back(std::make_pair(measurementName, (now - previous).count()));
                previous = providentia::utils::TimeMeasurable::now();
            }

            long getTotalMilliseconds() {
                return (timestamps[timestamps.size() - 1].second - timestamps[0].second).count();
            }

            std::string durations_str() {
                std::stringstream ss;

                for (const auto &duration : durations) {
                    ss << "[" << name << "] " << duration.second << ": " << duration.first << std::endl;
                }

                return ss.str();
            }
        };
    }
}

#endif //DYNAMICSTABILIZATION_UTILS_HPP
