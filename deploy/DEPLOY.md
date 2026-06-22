# Деплой на Raspberry Pi 🍓

Цель — автономный акустический пост: Raspberry Pi + USB-микрофон,
который слушает небо 24/7 и поднимает тревогу. «Датчик за ~$35».

Подойдёт Raspberry Pi 4 (2 ГБ+) или Pi 5. CNN маленькая — крутится на CPU
в реальном времени.

## 1. Система и звук

```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2 libsndfile1
# проверь, что микрофон виден:
arecord -l
```

## 2. Проект и зависимости

```bash
git clone https://github.com/lastlofty/drone-acoustic-detector.git
cd drone-acoustic-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> На ARM ставь **CPU-сборку PyTorch** (по умолчанию pip её и поставит).
> GPU на Pi не нужен — модель лёгкая.

## 3. Модель

Скопируй обученную модель `models/drone_cnn.pt` с рабочей машины
(или обучи прямо на Pi: `cd src && python make_synthetic_data.py && python train.py`).

> Для боевого качества модель должна быть обучена на **реальном**
> датасете, а не на синтетике. См. основной README.

## 4. Проверка вручную

```bash
cd src
python detect.py        # терминал
# или веб-дашборд:
python webapp.py        # http://<ip-малинки>:8000
```

## 5. Автозапуск как сервис

```bash
sudo cp deploy/drone-detector.service /etc/systemd/system/
# отредактируй User и пути внутри файла под свою систему!
sudo systemctl daemon-reload
sudo systemctl enable --now drone-detector
sudo systemctl status drone-detector
journalctl -u drone-detector -f      # логи в реальном времени
```

## 6. Идеи для «поста»

- **Buzzer/светодиод на GPIO** — физическая тревога при детекте.
- **Telegram-бот** — слать уведомление с вероятностью и временем.
- **Несколько Pi по периметру** → разница времени прихода звука даёт
  грубое направление на источник (триангуляция).

## Замечание

Это учебный/исследовательский пост наблюдения. Развёртывание реальных
систем мониторинга регулируется законом — см. дисклеймер в основном README.
