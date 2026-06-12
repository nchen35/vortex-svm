ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bfs_csr

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n 64

KERNEL_LIB := vortex2

# Runs under the same VM-enabled stack as the SVM tests (fair TLB comparison).
#   make TARGET=simx CONFIGS="-DVM_ENABLE"
CONFIGS ?= -DVM_ENABLE

include ../common.mk
