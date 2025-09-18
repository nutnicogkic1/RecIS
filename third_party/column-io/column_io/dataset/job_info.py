# -*- coding: utf-8 -*-
# This file is part of the PAI-XDL project. Used to get run time arg&config of the full task
import os,sys,time,json,argparse,uuid

class JobConfigKey:
    UNKNOWN = "kUnknown"
    # nebula config key
    NEBULA_USER_ID = "_NEBULA_USER_ID"
    NEBULA_PROJECT = "NEBULA_PROJECT"
    SCHEDULER_QUEUE = "scheduler_queue"
    TASK_ID = "TASK_ID"
    APP_ID = "APP_ID"
    # general config key
    DOCKER_IMAGE = "docker_image"
    HALO_WORKER_DOCKER_IMAGE = "halo_worker_docker_image"
    USER_ID = "USER_ID"
    TASK_NAME = "TASK_NAME"
    TASK_INDEX = "TASK_INDEX"
    RANK = "RANK"

def get_app_config(): # TODO add cache variable
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument('--config', default=None)
    parser.add_argument('--app_id', default=None)
    args, unknown = parser.parse_known_args()
    app_config = json.load(open(args.config)) if args.config else {}
    app_id = args.app_id if args.app_id else os.getenv(JobConfigKey.APP_ID, None)
    return app_config, app_id

def get_work_id():
    # mdl style
    worker_id = os.environ.get(JobConfigKey.RANK, None)
    if worker_id is not None:
        return int(worker_id)
    # xdl style
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--task_index", default=None)
    # parser.add_argument("--zk_addr", default=None) # some occasion use coreDNS, not a universal solution
    args, unknown = parser.parse_known_args()
    if args.task_index is None: # and args.zk_addr is None:
        os.environ[JobConfigKey.RANK] = str(0)
        return 0
    # assert args.task_index, "task_index can't be None in distributed mode"
    task_index = int(args.task_index)
    os.environ[JobConfigKey.RANK] = str(task_index)
    return task_index

def get_app_id(worker_id = get_work_id()):
    app_id = str(os.environ.get(JobConfigKey.APP_ID, "")) # both mdl and xdl style contains this env
    if app_id != "":
        return app_id
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--app_id", default=None)
    args, unknown = parser.parse_known_args()
    app_id = args.app_id if args.app_id else str(uuid.uuid4()).replace("-", "")[:12]
    app_id = (
        "local_"
        + app_id
        + "_"
        + str(worker_id)
        + "_"
        + str(uuid.uuid4()).replace("-", "")[:12]
    )
    return app_id

def get_task_name():
    task_name = os.environ.get(JobConfigKey.TASK_NAME, "")
    if task_name != "":
        return task_name
    parser = argparse.ArgumentParser(description="xdl arguments")
    parser.add_argument("--task_name", default=None)
    args, unknown = parser.parse_known_args()
    task_name = str(args.task_name) if args.task_name else "worker"
    os.environ[JobConfigKey.TASK_NAME] = task_name
    return task_name


# all_part_set_len = None
# def get_all_part_set_len_from_config():
#     global all_part_set_len
#     if all_part_set_len:
#         return all_part_set_len
#     config, app_id = get_app_config()
#     tables = str(config.get("odps_table", None))
#     if tables == "" or tables == "None":
#         all_part_set_len = 1
#         return all_part_set_len
#     tables = str(tables)
#     all_part_set_len = len(tables.split(","))
#     return all_part_set_len
__odps_table = None
def get_odps_table():
    # type: () -> str
    global __odps_table
    if __odps_table: # maybe "", also allowed
        return __odps_table
    config, app_id = get_app_config()
    __odps_table = str(config.get("odps_table", ""))
    return __odps_table

class JobInfo(object):
    def __init__(self, user_id, task_id, task_name, app_id, nebula_project=JobConfigKey.UNKNOWN, 
        scheduler_queue=JobConfigKey.UNKNOWN, is_foreign=0, rank=get_work_id(), 
        docker_image=JobConfigKey.UNKNOWN, halo_worker_docker_image=JobConfigKey.UNKNOWN):
        self._user_id = user_id
        self._task_id = task_id
        self._task_name = task_name
        self._app_id = app_id
        self._nebula_project = nebula_project
        self._scheduler_queue = scheduler_queue
        self._docker_image = docker_image
        self._is_foreign = is_foreign
        self._rank = rank
        self._halo_worker_docker_image = halo_worker_docker_image

def get_job_info():
    app_config, app_id = get_app_config()
    task_name = get_task_name()
    task_id = os.getenv(JobConfigKey.TASK_ID, None)
    user_id = app_config.get(JobConfigKey.NEBULA_USER_ID, os.getenv(JobConfigKey.USER_ID, JobConfigKey.UNKNOWN))
    nebula_project = os.getenv(JobConfigKey.NEBULA_PROJECT, JobConfigKey.UNKNOWN)
    scheduler_queue = app_config.get(JobConfigKey.SCHEDULER_QUEUE, JobConfigKey.UNKNOWN)
    docker_image = app_config.get(JobConfigKey.DOCKER_IMAGE, JobConfigKey.UNKNOWN)
    halo_worker_docker_image = app_config.get(JobConfigKey.HALO_WORKER_DOCKER_IMAGE, JobConfigKey.UNKNOWN)
    # worker_id = os.getenv(JobConfigKey.TASK_INDEX, 0) # actully it is pod INDEX, not worker-process index
    job_info = JobInfo(user_id=user_id, task_id=task_id, task_name=task_name,app_id=app_id,
                    nebula_project=nebula_project, scheduler_queue=scheduler_queue,
                    docker_image=docker_image, halo_worker_docker_image=halo_worker_docker_image)
    return job_info

def is_nootbook():
    return str(os.environ.get("NOTEBOOK_CONTAINER", "0")) == "1"