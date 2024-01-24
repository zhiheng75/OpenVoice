
LOG_DIR=logs

source venv/bin/activate
export PYTHONPATH=/home/zhihengw/staging/OpenVoice:${PYTHONPATH}

nohup python -m openvoice_app >"${LOG_DIR}"/openvoice.nohup 2>&1 &
