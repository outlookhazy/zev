CARGO ?= cargo
PORT  ?= /dev/ttyACM0
BUILD_DIR ?= $(or $(TMPDIR),/tmp)/zev-build
ZEVC  := $(CARGO) run --quiet --bin zevc --

.PHONY: all build doc check test ram demo qemu esp32c6 flash-esp32c6 uno flash-uno nano-every flash-nano-every pico flash-pico nucleo-f334r8 flash-nucleo-f334r8

all: test

build:
	$(CARGO) build --bin zevc

doc:
	$(ZEVC) doc >/dev/null
	$(ZEVC) doc --list >/dev/null
	@for q in syntax errors tasks yielding primitives dma runtime targets P0001 P0010 P0011 P0012 P0013 P0014 E0001 E0008 E0017 E0024 E0025 E0026 E0027 E0028 E0029 E0030 E0031 E0032 E0033 E0034; do \
		$(ZEVC) doc $$q >/dev/null || exit 1; \
	done
	@echo "OK (zevc doc primer, 8 topics, and error explainers verified)"

check:
	$(ZEVC) check examples/motor_controller.zev examples/motor_support.zev
	$(ZEVC) check examples/sim_demo.zev
	$(ZEVC) check examples/cortex_m_demo.zev
	$(ZEVC) check examples/esp32c6_demo.zev
	$(ZEVC) check examples/arduino_uno_demo.zev
	$(ZEVC) check examples/arduino_nano_every_demo.zev
	$(ZEVC) check examples/pico_w_demo.zev
	$(ZEVC) check examples/portable_controller_demo.zev
	$(ZEVC) check --board lm3s6965_evb examples/portable_controller_demo.zev
	$(ZEVC) check --board nucleo_f334r8 examples/portable_controller_demo.zev
	$(ZEVC) check --board esp32c6_pico examples/portable_controller_demo.zev
	$(ZEVC) check --board rpi_pico_w examples/portable_controller_demo.zev
	$(ZEVC) check --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev

ram:
	$(ZEVC) ram examples/motor_controller.zev examples/motor_support.zev

demo:
	$(ZEVC) run --sanitize examples/sim_demo.zev

qemu:
	$(ZEVC) run-qemu examples/cortex_m_demo.zev

esp32c6:
	$(ZEVC) build --board esp32c6_pico examples/esp32c6_demo.zev -o $(BUILD_DIR)/esp32c6/esp32c6_demo.bin

flash-esp32c6:
	$(ZEVC) flash --board esp32c6_pico examples/esp32c6_demo.zev --port $(PORT)

uno:
	$(ZEVC) build --board arduino_uno examples/arduino_uno_demo.zev -o $(BUILD_DIR)/avr/arduino_uno_demo.bin

flash-uno:
	$(ZEVC) flash --board arduino_uno examples/arduino_uno_demo.zev --port $(PORT)

nano-every:
	$(ZEVC) build --board arduino_nano_every examples/arduino_nano_every_demo.zev -o $(BUILD_DIR)/nano-every/arduino_nano_every_demo.bin

flash-nano-every:
	$(ZEVC) flash --board arduino_nano_every examples/arduino_nano_every_demo.zev --port $(PORT)

pico:
	$(ZEVC) build --board rpi_pico_w examples/pico_w_demo.zev -o $(BUILD_DIR)/rp2040/pico_w_demo.uf2

flash-pico:
	$(ZEVC) flash --board rpi_pico_w examples/pico_w_demo.zev --port $(PORT)

nucleo-f334r8:
	$(ZEVC) build --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev -o $(BUILD_DIR)/f334r8/nucleo_f334r8_demo.bin

flash-nucleo-f334r8:
	$(ZEVC) flash --board nucleo_f334r8 examples/nucleo_f334r8_demo.zev --port $(PORT)

test:
	$(CARGO) test --workspace -- --test-threads=2
