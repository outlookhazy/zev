CARGO ?= cargo
PORT  ?= /dev/ttyACM0
ZEVC  := $(CARGO) run --quiet --bin zevc --

.PHONY: all build doc check test ram demo qemu esp32c6 flash-esp32c6

all: test

build:
	$(CARGO) build --bin zevc

doc:
	$(ZEVC) doc >/dev/null
	$(ZEVC) doc --list >/dev/null
	@for q in syntax errors tasks yielding primitives dma runtime targets P0001 P0010 P0011 P0012 E0001 E0008 E0017 E0024 E0025 E0026 E0027 E0028 E0029; do \
		$(ZEVC) doc $$q >/dev/null || exit 1; \
	done
	@echo "OK (zevc doc primer, 8 topics, and error explainers verified)"

check:
	$(ZEVC) check examples/motor_controller.zev examples/motor_support.zev
	$(ZEVC) check examples/sim_demo.zev
	$(ZEVC) check examples/cortex_m_demo.zev
	$(ZEVC) check examples/esp32c6_demo.zev

ram:
	$(ZEVC) ram examples/motor_controller.zev examples/motor_support.zev

demo:
	$(ZEVC) run --sanitize examples/sim_demo.zev

qemu:
	$(ZEVC) run-qemu examples/cortex_m_demo.zev

esp32c6:
	$(ZEVC) build-esp32c6 examples/esp32c6_demo.zev -o /tmp/zev-esp32c6/esp32c6_demo.bin

flash-esp32c6:
	$(ZEVC) flash-esp32c6 examples/esp32c6_demo.zev --port $(PORT)

test:
	$(CARGO) test --workspace -- --test-threads=2
