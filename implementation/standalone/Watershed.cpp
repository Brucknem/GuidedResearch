//
// Created by brucknem on 13.01.21.
//
#include "Watersheder.hpp"

/**
 * Standalone mode for the watersheder.
 */
int main(int argc, char const *argv[]) {
	auto watersheder = providentia::calibration::Watersheder(
		"../misc/test_frame.png"
	);
	watersheder.run();

	return 0;
}
