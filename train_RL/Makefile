# Create carla sif file if it does not exist
CARLA_VERSION=0.9.13
CARLA_SIF = "carla-$(CARLA_VERSION).sif"
CARLA_DEF = "./carla_server.def"

TRAINER_SIF = "trainer.sif"
TRAINER_DEF = "./trainer.def"


CARLA_SERVER_PORT?=2000
CARLA_SERVER_GPU_DEVICE?=0

PORTS?=2000
TRAFFIC_MANAGER_PORTS?=8000

build-carla:
	if [ ! -f "$(CARLA_SIF)" ]; then \
		apptainer build $(CARLA_SIF) $(CARLA_DEF); \
	fi


run-carla: build-carla
	while true; do \
	apptainer exec --nv $(CARLA_SIF) /home/carla/CarlaUE4.sh --world-port=$(CARLA_SERVER_PORT) -RenderOffScreen \
	-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=$(CARLA_SERVER_GPU_DEVICE) && break; \
  sleep 1; \
	done



build-trainer:
	if [ ! -f "$(TRAINER_SIF)" ]; then \
		mkdir /tmp/haavasma_trainer_package; \
		rsync -av --exclude-from='../.gitignore' ./ /tmp/haavasma_trainer_package/; \
		apptainer build $(TRAINER_SIF) $(TRAINER_DEF); \
	fi


train: build-trainer 
	apptainer exec --nv -B ./:/app $(TRAINER_SIF) bash -c "\
		. /venv/bin/activate && \
    cd /app && \
		python train_rl_vanilla.py --ports=$(PORTS) --traffic-manager-ports=$(TRAFFIC_MANAGER_PORTS) \
"

