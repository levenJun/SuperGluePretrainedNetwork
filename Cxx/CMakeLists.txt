cmake_minimum_required(VERSION 3.10)
project(TestSpCxx)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
if((CMAKE_BUILD_TYPE MATCHES "Release") OR (CMAKE_BUILD_TYPE MATCHES  "RelWithDebInfo"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -O3")
	# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -fPIC -g -fstack-protector -fsanitize=address -fno-omit-frame-pointer")
else() # Debug or MinSizeRel
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -fPIC -O0 -g -fstack-protector -fsanitize=address -fno-omit-frame-pointer")
    add_definitions(-DBUILD_DEBUG)
endif()

add_definitions(-DCOMPILEDWITHC11)

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy)
IF(Python3_FOUND)
    MESSAGE(STATUS "Found Python3")
	#需要pip安装:pip install pybind11
	#然后在python环境下 执行 import pybind11 和 print(pybind11.get_cmake_dir()) 获取pybind11的cmake路径并按下面方式设置进来
	set(pybind11_DIR /home/delljun/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
    FIND_PACKAGE(pybind11 REQUIRED)		
ENDIF(Python3_FOUND)

IF(Python3_FOUND)
	SET(PUBLIC_LIBRARIES
		${PUBLIC_LIBRARIES}
		Python3::Python
		Python3::NumPy
	)
	SET(LIBRARIES
		${LIBRARIES}
		pybind11::embed
	)
	SET(INCLUDE_DIRS
		${INCLUDE_DIRS}
	)
ENDIF(Python3_FOUND)

find_package(OpenCV 3.4)
	if(NOT OpenCV_FOUND)
      	message(FATAL_ERROR "OpenCV > 3.4 not found.")
	else()
		SET(PUBLIC_LIBRARIES
			${PUBLIC_LIBRARIES}
			${OpenCV_LIBS}
		)
   	endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

file(GLOB UTILS_SRC "Interpreter/Utils/*.cpp")
file(GLOB SUPERPOINT_SRC "Interpreter/Python/*.cpp")

add_executable(${PROJECT_NAME} test/TestSpCxx.cpp ${UTILS_SRC} ${SUPERPOINT_SRC})

# Link against Python libraries
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ./ )
target_link_libraries(${PROJECT_NAME} PRIVATE ${PUBLIC_LIBRARIES} ${LIBRARIES})