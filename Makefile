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

test-yolo:
	python detect\yolov5\detect.py --weights detect\yolov5\best.pt --img 640 --conf 0.25 --source test\data\phone.mp4 --save-txt

sample-bg:
	python samples\bg_sub.py --input test\data\hik-full2.mp4 --save test\results\bg.mp4

sample-mark:
	python samples\mark.py --input test\data\hik-full2.mp4
# $env:PYTHONPATH += ';'+(pwd)