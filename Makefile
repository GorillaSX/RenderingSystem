rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

UNAME := $(shell uname -s | tr "[:upper:]" "[:lower:]")
SOURCES := $(call rwildcard, src/, *.cpp)
OBJS := $(subst src/,build/,$(SOURCES:.cpp=.o))
CU_SOURCES := $(call rwildcard, src/, *.cu)
ALLOBJ = $(call rwildcard, build/, *.o)
CU_OBJS := $(subst src/,build/,$(CU_SOURCES:.cu=.cu.o))
CFLAGS = -isystem include -Isrc -I/usr/local/include/opencv -I/usr/local/include
TARGET = Gorilla


LDFLAGS = -lGL -lglfw  -lboost_system -lboost_filesystem -lboost_program_options  -lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_photo -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_cudaarithm -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core -lopencv_cudev


CXX = nvcc
CFLAGS := --std=c++11 --machine 64 --gpu-architecture=sm_52 --use_fast_math -dc --cudart static -Xcompiler "$(CFLAGS)"
LINKFLAG = --gpu-architecture=sm_52 -std=c++11 -DGLEW_STATIC --cudart static -Xcompiler -isystem  -Isrc 

default: Gorilla

Gorilla: $(OBJS) $(CU_OBJS)
	@mkdir -p bin
	@echo "Linking $@"
	@$(CXX) $(OBJS) $(CU_OBJS) $(LINKFLAG) $(LDFLAGS) -o bin/$(TARGET)

build/%.o: src/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling $<"
	@$(CXX) $(CFLAGS) -c -o $@ $<

build/%.cu.o: src/%.cu
	@mkdir -p $(@D)
	@echo "Compiling $<"
	@$(CXX) $(CFLAGS) -c -o $@ $<

clean:
	@rm -rf bin build
