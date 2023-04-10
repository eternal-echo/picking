.PHONY: run-server run-client

run-server:
	# python -u server_simulator.py --port "COM1"
	python -u examples\server_sync.py --comm "serial" --framer "rtu" --log "debug" --port "COM1" --baudrate 9600

run-client:
	python examples\client_sync.py --comm "serial" --framer "rtu" --log "debug" --port "COM2" --baudrate 9600

test-unit:
	pytest -m unit

test-inte:
	pytest -m integration
