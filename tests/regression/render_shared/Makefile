ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := render_shared

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

# Single-backing-store SVM requires the VM subsystem.
#   make TARGET=simx CONFIGS="-DVM_ENABLE"
CONFIGS ?= -DVM_ENABLE

include ../common.mk
