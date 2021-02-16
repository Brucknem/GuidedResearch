cmake_minimum_required(VERSION 3.10)

file(GLOB IMAGES "*.png")
file(COPY ${IMAGES} DESTINATION ./)

file(GLOB OPENDRIVE "*.xodr")
file(COPY ${OPENDRIVE} DESTINATION ./)