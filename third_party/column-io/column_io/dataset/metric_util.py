# -*- coding: utf-8 -*-
import os,sys,time,datetime
import threading
import pykmonitor # TODO: reimport pykmonitor if need
from column_io.dataset.job_info import get_work_id,get_job_info

class MetricStatus():
    # expected status
    SUCCESS = 0 # ok
    WAITING = 1001 # waiting status

    # cross-processes  errors
    REQUEST_ERROR = 2001 # http, rpc request error
    CALL_ERROR = 2002 # ipc, internal-func call error
    LOCAL_CACHE_ERROR = 2003 # local cache error
    LOCAL_CACHE_FILE_NOT_EXISTS_ERROR = 2004 # local cache file not exists

    # logic error in process
    JSON_ERROR = 3001 #
    CODEC_ERROR = 3002 # e.g. utf-8 error
    FIELD_ERROR = 3003 # field missing, type-wrong or other relative error
    OUTDATE_ERROR = 3011  # context(session, token,,,) is expired

    # un classified errors
    UNKNOWN_ERROR = 9999

    pass

# This Exception aims to replace tf.errors or absl error so as to re-used in columnio
# Strictly speaking, this should be an independent Python module or file. 
# Also, check if tf/abseil errors provides other useful actions()
# However, in order to quickly launch, it is temporarily placed here :)
class NebulaIOFatalError(Exception):
    def __init__(self, msg):        
        self.msg = msg
        # metric report
        self._report_metric()
    def _report_metric(self):
        global metric_factory
        metric_client = metric_factory.get("openstorage_session_init")
        metric_client.try_start()
        metric_tag_map = {
            "code": "1",
            "status": "fail",
        }
        metric_client.report(MetricPoints.init_qps, 1, metric_tag_map)
        time.sleep(1.5)
        # metric_client.try_close()
    def __str__(self):
        # return "NebulaIOFatalError: {}".format(self.msg)
        return self.msg

class MetricPoints():
    class Point():
        def __init__(self, name, metric_type=pykmonitor.MetricType.GAUGE, priority=pykmonitor.PriorityType.MAJOR, statistics_type=None):
            self.name = name # type: str # point name 
            self.metric_type = metric_type # type: pykmonitor.MetricType # the point static type 
            self.priority = priority # type: pykmonitor.PriorityType # the point static period level
            self.statistics_type = statistics_type # type: pykmonitor.StatisticsType # the point static statistics type

    # TODO: v1/v2 prefix in future if need
    # GAUGE
    # E.g. xdl.metric.openstorage_session_refresh.latency_ipc_ms/xdl.metric.openstorage_session_cache.cache_qps
    latency_get_ms = Point("latency_get_ms", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.AVG)
    latency_post_ms = Point("latency_post_ms", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.AVG)
    latency_ipc_ms = Point("latency_ipc_ms", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.AVG)
    
    cache_qps = Point("cache_qps", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.SUM)
    create_qps = Point("create_qps", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.SUM)
    refresh_qps = Point("refresh_qps", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.SUM)

    init_qps = Point("init_qps", metric_type=pykmonitor.MetricType.GAUGE, statistics_type=pykmonitor.StatisticsType.SUM)

    # COUNTER
    pass
    # QPS
    pass

    @staticmethod
    def _get_points():
        # type: () -> list[Point] # return all available points
        return [
            MetricPoints.latency_get_ms, # for cache get
            MetricPoints.latency_post_ms, # for post new session cache
            MetricPoints.latency_ipc_ms, # for refresh call with ipc halo-worker
            MetricPoints.cache_qps,
            MetricPoints.create_qps,
            MetricPoints.refresh_qps,
            MetricPoints.init_qps,
        ]
  
    @staticmethod
    def regist_all_metric_points(kmonitor):
        # type: (pykmonitor.KMonitor) -> None
        for p in MetricPoints._get_points():
            if p.statistics_type:
                kmonitor.register(p.name, p.metric_type, p.priority, p.statistics_type)
            else:
                kmonitor.register(p.name, p.metric_type, p.priority)
            # logger.debug("successfully regist metric {}".format(name))


class KMonitorClient():
    def __init__(self, namespace, module, tenant, global_tag_map):
        # self._metric_namespace = namespace
        self._status_mutex = threading.Lock()
        self._metric_module = module
        self._global_tag_map = global_tag_map #type: dict
        # self._tenant = tenant
        host_ip = global_tag_map["host_ip"]
        host_port = global_tag_map["host_port"]
        host_ip = "[{}]".format(host_ip) if ":" in host_ip else host_ip
        sink_address = "{}:{}".format(host_ip, host_port)
        # print("[{}] [INFO] start init pykmonitor.KMonitorFactory, service_name:{}, sink_address:{}, global_map count:{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), namespace, sink_address, len(global_tag_map)), file=sys.stderr)
        pykmonitor.KMonitorConfig.set_sink_address(sink_address) # done but need fix in pykmonitor.KMonitor.__init__
        pykmonitor.KMonitorConfig.set_service_name(namespace)
        self._kmonitor = pykmonitor.KMonitor(self._metric_module) #type: pykmonitor.KMonitor 
        # self._kmonitor = KMonitorFactory.get_kmonitor(self._metric_module) #type: pykmonitor.KMonitor # cannot destroy in factory when close
        MetricPoints.regist_all_metric_points(self._kmonitor)


    def try_start(self):
        if self._kmonitor and self._kmonitor.is_alive():
            return
        with self._status_mutex:
            self._kmonitor = pykmonitor.KMonitor(self._metric_module) #type: pykmonitor.KMonitor 
            MetricPoints.regist_all_metric_points(self._kmonitor)

    def running(self):
        with self._status_mutex:
            return self._kmonitor and self._kmonitor.is_alive()

    def try_close(self):
        # print("[{}] [DEBUG] try closeing kmonitor metric for {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), self._metric_module))
        time.sleep(1.5) # wait for metric agent report
        with self._status_mutex:
            if self._kmonitor and self._kmonitor.is_alive():
                self._kmonitor.close()
                # print("[{}] [DEBUG] try closed kmonitor metric for {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), self._metric_module))
        # print("[{}] [DEBUG] try close done kmonitor metric for {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), self._metric_module))

    def _build_tags(self, tag_map):
        if len(tag_map) == 0: # for faster
            all_tag_map = self._global_tag_map
        else:
            all_tag_map = {}
            all_tag_map.update(self._global_tag_map)
            all_tag_map.update(tag_map)
        return pykmonitor.ImmutableMetricTags(all_tag_map)

    """
    Args:
        metric_name: metric point name, must regist first, full name is {namespace}.{module}.{metric_name}, DO NOT use char except a-z A-Z 0-9 . - _
        metric_value: metric point value, must be numeric.
        tag_map: dict format {tag_name: tag_value}
    """
    def report(self, point, metric_value, tag_map={}):
        # type: (MetricPoints.Point, float, dict) -> None
        if not self._kmonitor.is_alive():
            # print("[{}] [WARN] kmonitor is not running, skip report".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
            return
        # if not re.match("^[a-zA-Z0-9\.\-_]+$", metric_name):
        #   raise ValueError("metric_name {} contains char other than a-z A-Z 0-9 . - _".format(metric_name))
        tags = self._build_tags(tag_map)
        # print("[{}] [INFO] report kmonitor metric name:{}, value:{}, tag_map:{}, tags:{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f"), self._metric_module+"."+point.name, metric_value, tag_map, tags))
        self._kmonitor.report(metric_name=point.name, immutable_metrics_tags=tags, value=metric_value, )


_k8s_node_ip = None
def get_k8s_node_ip():
    # type: () -> str
    global _k8s_node_ip
    if _k8s_node_ip:
        return _k8s_node_ip
    def get_k8s_node_ip_internal():
        # 1. if env "RequestedIP" and is ipv6, then use /etc/hostinfo-ipv6
        if os.environ.get("RequestedIP") and ":" in os.environ.get("RequestedIP") and os.path.exists("/etc/hostinfo-ipv6"):
            # read file content of /etc/hostinfo-ipv6, multi-line like: "tre-3-a100-535-0035\n33.103.193.157"
            with open("/etc/hostinfo-ipv6", "r") as f:
                for line in f:
                    if ":" in line:
                        return str(line.strip()) # line is ipv6
        # 2. if env "RequestedIP" not exist, or it is ipv6, then use /etc/hostinfo
        if os.path.exists("/etc/hostinfo"):
            with open("/etc/hostinfo", "r") as f:
                for line in f:
                    if line.count('.') == 3 and all(0<=int(num)<256 for num in line.rstrip().split('.')):
                        return str(line.strip()) # line is ipv4
        # 3. use env "KUBERNETES_NODE_IP"
        if os.environ.get("KUBERNETES_NODE_IP"):
            return os.environ.get("KUBERNETES_NODE_IP")
        # 4. if all above not exist, then use "localhost"
        return "localhost"
    # IPV6策略尚有问题
    # _k8s_node_ip = get_k8s_node_ip_internal()
    _k8s_node_ip = os.environ.get("KUBERNETES_NODE_IP", "localhost")

    # KMONITOR_SINK_ADDRESS tell c++ to get new style NODE_IP. C++ no need to imple again(also more complex)
    os.environ["KMONITOR_SINK_ADDRESS"] = _k8s_node_ip
    return _k8s_node_ip


class KMonitorClientFactory():
    job_info = get_job_info()
    metric_kwargs = {
        "namespace": "xdl.metrics",
        # "module": "openstorage",
        "tenant": "default",
        "global_tag_map": {
            "io_type": __name__.split('.')[0], # e.g. paiio/column/common => paiio,
            "host_ip": get_k8s_node_ip(),
            "host_port": 4141,
            "task_id": str(job_info._task_id),
            "app_id": str(job_info._app_id),
            "user_id": str(job_info._user_id),
            "task_name": str(job_info._task_name),
            "rank": str(job_info._rank),
            "docker_image": str(job_info._docker_image).split(":")[-1],
            "calculate_cluster": str(os.environ.get("CALCULATE_CLUSTER", "null")),
            "nebula_project": str(os.environ.get("NEBULA_PROJECT", "null")),
            "scheduler_queue": str(os.environ.get("SCHEDULER_QUEUE", "null")),
            "openstorage_backend": str(os.environ.get("OPEN_STORAGE_BACKEND", "null")),
            "tunnel_endpoint": str(os.environ.get("OPEN_STORAGE_TUNNEL_ENDPOINT", "null")),
            "sigma_app_site": str(os.environ.get("SIGMA_APP_SITE", "null")),
            # "ip": pod_ip, # auto added by kmonitor sdk
        },
    }
    def __init__(self):
        self._client_map = {} # type: dict[str, KMonitorClient] # {module: KMonitorClient}
        self._mutex = threading.Lock()
    def get(self, module):
        # type : (str) -> KMonitorClient
        if module in self._client_map:
            return self._client_map[module]
        with self._mutex:
            if module in self._client_map:
                return self._client_map[module]
            kwargs = KMonitorClientFactory.metric_kwargs
            self._client_map[module] = KMonitorClient(
                namespace=kwargs["namespace"],
                module=module,
                tenant=kwargs["tenant"],
                global_tag_map=kwargs["global_tag_map"],
            )
            return self._client_map[module]

metric_factory = KMonitorClientFactory()




def test_metric_point(times = 5):
    metric_client = metric_factory.get("openstorage_session_cache")
    metric_client.try_start()

    appid = os.environ.get("APP_ID", "xdl-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M")) )
    print("\t\t metric_client submit for {} begin...".format(appid))

    metric_client.report(MetricPoints.cache_qps, 1, { "app_id": appid } )
    time.sleep(1.5)
    print("[{}] [INFO] metric_client closing...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))
    metric_client.try_close()
    print("[{}] [INFO] metric_client close done".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")))

# Test me if need. I work fine
# if __name__ == "__main__":
#     test_metric_point()

