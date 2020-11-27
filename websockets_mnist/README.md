# Federated learning using websockets - MNIST example

The scripts in this folder let you execute a federated training via three websocket connections.

The script start_websocket_servers.py will start the Websocket server workers for Alice, Bob and Charlie.
```
$ python start_websocket_servers.py
```

The training is then started by running the script run_websocket_client.py:
```
$ python run_websocket_client.py
```
This script
 * loads the MNIST dataset,
 * distributes it onto the three workers
 * starts a federated training.

 The federated training loop contains the following steps
 * the current model is sent to the workers
 * the workers train on a fixed number of batches
 * the three models from Alice, Bob and Charlie are then averaged (federated averaging)

 This training loop is then executed for a given number of epochs.
 The performance on the test set of MNIST is shown after each epoch.


# README

마지막 수정 일자: 2020년 11월 2일
상태: 📝게시판
작성일시: 2020년 10월 5일 오전 10:48

[사용한 백그라운드(RPi & 중앙디바이스)](https://www.notion.so/427c63c9166a4cf5bbd43fa5e0b343ff)

[준비할 것](https://www.notion.so/3c23991ea95f456180a024c9302a6c8f)

💡 **run_websocket_server(0.2.3).py**와 **run_websocket_client(0.2.3).py**에 대한 자세한 설명은 '*[이 곳](https://www.notion.so/CODE-GUIDE-8d038e9860324753a80dc4bfd4a88bd5)'*을 참조하십시오.

## 라즈베리파이에서의 서버 세팅 방법

1. **도커 이미지를 받아오지 않았을 경우(즉, 최초로 실행할 경우)**, 커맨드 창에서 다음의 명령어를 사용하여 이미지를 다운로드 합니다.

```jsx
docker pull lyb35548/syft-for-rpi:1.0.6
```

2. **도커 이미지가 이미 다운로드되어있고 컨테이너를 아직 만들지 않은 경우**, 커맨드 창에서 다음의 명령어를 실행하여 컨테이너를 생성 및 실행합니다. 이를 완료했다면 **[4로 건너뜁니다]()**. 

```jsx
docker run -t -i -p 10002:10002 -v /home:/home/workspace --ip 127.0.0.10 737509875171
```

💡이 문서에서 사용하는 이미지의 아이디는 **737509875171** 입니다. 생성된 컨테이너를 확인하고자 할때 이미지 아이디가 필요합니다.

3-1. **컨테이너를 이전에 만든 적이 있는 경우** , 커맨드 창에서 다음의 명령어를 실행하여 컨테이너 아이디를 확인합니다. 만들어진 컨테이너를 다시 사용하면 컨테이너를 추가적으로 만들지 않아도 됩니다. 이는 기기 메모리 관리에 도움을 줄 수 있기 때문에 권장되는 방식입니다. . 

```jsx
docker ps -a
```

3-2. 3-1을 실행할 경우 결과값의 **첫번째 열에 컨테이너의 아이디가 표시됩니다**. 두번째 열에서 컨테이너의 이미지 파일 아이디를 확인한 후 사용하고자 하는 이미지 파일 아이디와 일치하는 컨테이너를 고르십시오. 

```
CONTAINER ID IMAGE         COMMAND    CREATED      STATUS                    PORTS                               NAMES
33261d0d7f4a 737509875171 "/bin/bash" 23 hours ago Exited (255) 22 hours ago 10001/tcp, 0.0.0.0:10002->10002/tcp ecstatic_colden
68963f0781a5 3028d43a61ae "/bin/bash" 25 hours ago Exited (0) 23 hours ago
```

💡위 예시의 경우 사용하고자 하는 이미지 파일 아이디는 **737509875171**므로 컨테이너 아이디는 **33261d0d7f4a**에 해당합니다. 

3-3.

컨테이너 아이디를 확인한 경우, 커맨드 창에서 다음의 명령어를 실행하여 컨테이너를 시작 및 접속합니다. <컨테이너 아이디>에는 앞서 확인한 아이디를 대입합니다. 

```
docker start <컨테이너 아이디>
docerk attach <컨테이너 아이디>
```

💡 3-2에서 확인한 아이디를 예로 든다면 **docker start 33261d0d7f4a**가 됩니다. 

4. 도커 컨테이너에 진입하였다면 다음 명령어들을 사용하여 서버를 구동합니다.

```jsx
cd /home
python3 'run_websocket_server(0.2.3).py' --port 10002 --host 0.0.0.0 --id <아이디>
```

💡 <아이디>는 변경 가능하나 대개 **alice, bob, charlie,...**를 사용합니다. <아이디>를 바꿀 경우 중앙 디바이스의 코드 또한 바꿔주어야 합니다.

5. 서버가 정상적으로 동작할 경우 다음과 같은 문구가 표시됩니다. 

```jsx
Serving. Press CTRL-C to stop.
```

### 번외) 라즈베리파이에서 시작프로그램 설정하기

라즈베리파이를 켤 때마다 도커 컨테이너를 실행하는 수고를 덜기 위해 부팅 시 자동으로 컨테이너를 실행하게끔 설정합니다. 컨테이너가 이미 존재할 경우에만 가능합니다. 

1. 터미널 창에서 다음 명령어를 사용하여 'autostart'를 실행합니다.

```jsx
sudo mousepad /etc/xdg/lxsession/LXDE-pi/autostart
```

정상적으로 실행하였을 경우 다음과 같은 텍스트를 볼 수 있습니다. 기기에 따라 약간의 차이가 있을 수 있습니다.

```jsx
@lxpanel --profile LXDE-pi
@pcmanfm --desktop --profile LXDE-pi
@xscreensaver -no-splash
```

2. 맨 아래에 다음과 같은 텍스트를 추가합니다.  컨테이너 아이디를 확인하는 법은 3-2에 설명되어있습니다. 

```jsx
lxterminal -e docker start <컨테이너 아이디>
lxterminal -e docker attach <컨테이너 아이디>
```

3. 이제 라즈베리파이를 부팅할 때 자동으로 컨테이너가 실행됩니다. 

## 중앙 디바이스에서의 서버 세팅 방법

1. **run_websocket_client(0.2.3).py** 를 준비합니다.

💡 작업이 라즈베리파이가 아닌 중앙 디바이스에서 이루어지고 있는지 확인하십시오. 

2. 연결하고자 하는 IP와 포트가 올바르게 설정되어있는지 확인하십시오. 기본적으로 IP는 **'192.168.0.52~54'**, 포트는 '**10002'**으로 설정되어있습니다. IP와 포트를 변경할 경우 라즈베리파이에서 컨테이너 설정을 변경해 주어야 합니다. 

3. 아이디가 라즈베리파이에서 설정한 아이디와 일치하는지 확인한 후 파일을 실행합니다. ID는 라즈베리파이에서의 서버를 시작할 때 설정한 ID여야 하며 기본적으로 alice, bob, charlie로 설정되어있습니다. 

4. 서버가 정상적으로 동작할 경우 다음과 같은 문구가 표시됩니다. 

```bash
2020-10-06 15:01:11,117 INFO [dataset.py](http://dataset.py/)(l:138) - Scanning and sending data to alice, bob, charlie...
2020-10-06 15:01:15,646 DEBUG [dataset.py](http://dataset.py/)(l:147) - Sending data to worker alice
2020-10-06 15:01:26,890 DEBUG [dataset.py](http://dataset.py/)(l:147) - Sending data to worker bob
2020-10-06 15:01:38,482 DEBUG [dataset.py](http://dataset.py/)(l:147) - Sending data to worker charlie
2020-10-06 15:01:45,389 DEBUG [dataset.py](http://dataset.py/)(l:152) - Done!
2020-10-06 15:01:45,409 INFO run_websocket_client(0.2.3).py(l:263) - Starting epoch 1/2
2020-10-06 15:04:32,882 DEBUG run_websocket_client(0.2.3).py(l:125) - Starting training round, batches [0, 50]
2020-10-06 15:04:36,346 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker alice: [0/50 (0%)]	Loss: 2.310694
2020-10-06 15:05:37,873 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker alice: [25/50 (50%)]	Loss: 2.204359
2020-10-06 15:06:43,192 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker bob: [0/50 (0%)]	Loss: 2.298535
2020-10-06 15:07:51,168 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker bob: [25/50 (50%)]	Loss: 2.222411
2020-10-06 15:09:17,090 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker charlie: [0/50 (0%)]	Loss: 2.314186
2020-10-06 15:10:36,904 DEBUG run_websocket_client(0.2.3).py(l:75) - Train Worker charlie: [25/50 (50%)]	Loss: 2.209582

이하생략...
```

## 흔히 일어나는 오류

### ❌ RuntimeWarning: coroutine 'WebsocketServerWorker._consumer_handler' was never awaited

**가능한 원인**

1. 라즈베리파이의 서버상태가 불량하여 접속이 끊어진 상태일 수 있습니다. 모든 라즈베리파이가 서버를 정상적으로 구동중인지 확인하십시오.
2. 중앙 디바이스에서 설정한 라즈베리파이 개수가 실제로 구동중인 라즈베리파이보다 많을 수 있습니다. 중앙 디바이스의 코드에서 설정한 개수와 구동중인 라즈베리파이 개수가 일치하는지 확인하십시오. 

❌ **File "/usr/local/lib/python3.7/site-packages/syft/serde/serde.py", line 543, in _detail
return detailers[obj[0]](worker, obj[1])
IndexError: list index out of range** 

*혹은* 

❌ **File "/usr/local/lib/python3.7/site-packages/syft/serde/msgpack/serde.py" line 460, in _detail
return detailers[obj[0]](worker, obj[1], **kwargs)
KeyError: 53**

**가능한 원인**

1. 중앙 디바이스와 라즈베리파이의 Pysyft 버전이 달라 일어나는 오류일 수 있습니다. 중앙 디바이스와 라즈베리파이의 Pysyft 버전이 일치하는지 확인하십시오. 다음 명령어를 커맨드창에 입력하여 Pysyft의 버전을 확인할 수 있습니다.

    ```bash
    pip show syft
    ```

2. 중앙 디바이스 혹은 라즈베리파이의 Pysyft 버전과 구동중인 'run_websocket_server.py' 혹은 'run_websocket_client.py'의 버전이 달라 일어나는 오류일 수 있습니다. 각 장치에서 Pysyft 버전과 py파일의 버전이 일치하는지 확인하십시오. 

    💡 기본적으로 모든 py파일의 이름에 버전을 병기하였으나 만약 버전을 확인할 수 없다면 [https://github.com/OpenMined/PySyft](https://github.com/OpenMined/PySyft) 에서 확인할 수 있습니다.
