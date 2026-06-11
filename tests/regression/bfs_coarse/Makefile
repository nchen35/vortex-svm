ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bfs_coarse

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n 64

KERNEL_LIB := vortex2

# SVM requires the VM subsystem. Build with:
#   make TARGET=simx CONFIGS="-DVM_ENABLE"
CONFIGS ?= -DVM_ENABLE

include ../common.mk
