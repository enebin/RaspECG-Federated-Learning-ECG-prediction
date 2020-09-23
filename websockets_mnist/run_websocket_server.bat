@echo on
call C:\Users\Lee\anaconda3\Scripts\activate.bat
call activate websockets_mnist

set SERVER_NO = 3

start python ./run_websocket_server.py --port 10001 --id alice
start python ./run_websocket_server.py --port 10002 --id bob
start python ./run_websocket_server.py --port 10003 --id charlie

cmd /k