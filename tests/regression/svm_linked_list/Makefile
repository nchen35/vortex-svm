ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := svm_linked_list

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n 32

KERNEL_LIB := vortex2

# SVM requires the VM subsystem. Build with:
#   make TARGET=simx CONFIGS="-DVM_ENABLE"
# or set VM_ENABLE = true in VX_config.toml before running configure.
CONFIGS ?= -DVM_ENABLE

include ../common.mk
