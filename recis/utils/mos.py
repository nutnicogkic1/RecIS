import os
import re
import time

from nebula.mos import ModelCkpt, ModelVersion

from recis.utils.logger import Logger


logger = Logger(__name__)


def parse_uri(uri, auto_version=False):
    """Parse and validate MOS URI format.

    This function parses a MOS URI string and extracts its components including
    project, model name, version, and checkpoint ID. It validates the URI format
    and optionally generates automatic version timestamps.

    Args:
        uri (str): MOS URI string in format:
            - "model.project.model_name" (model level)
            - "model.project.model_name/version=version_id" (version level)
            - "model.project.model_name/version=version_id/ckpt_id=checkpoint_id" (checkpoint level)
        auto_version (bool, optional): If True, automatically generates a timestamp-based
            version when URI is at model level. Defaults to False.

    Returns:
        dict: Dictionary containing parsed URI components:
            - uri (str): Original or modified URI
            - project (str): Project name
            - model_name (str): Model name
            - version (str or None): Version identifier
            - ckpt_id (str or None): Checkpoint identifier
            - level (str): URI level ('model', 'version', or 'ckpt_id')

    Raises:
        ValueError: If URI format is invalid.

    Example:
        >>> # Parse model-level URI
        >>> info = parse_uri("model.recommendation.user_embedding")
        >>> print(info["level"])  # 'model'
        >>> print(info["project"])  # 'recommendation'
        >>> # Parse version-level URI
        >>> info = parse_uri("model.recommendation.user_embedding/version=v1.0")
        >>> print(info["level"])  # 'version'
        >>> print(info["version"])  # 'v1.0'
        >>> # Parse checkpoint-level URI
        >>> uri = "model.recommendation.user_embedding/version=v1.0/ckpt_id=step_1000"
        >>> info = parse_uri(uri)
        >>> print(info["level"])  # 'ckpt_id'
        >>> print(info["ckpt_id"])  # 'step_1000'
        >>> # Auto-generate version
        >>> info = parse_uri("model.recommendation.user_embedding", auto_version=True)
        >>> print(info["version"])  # '20231201120000' (timestamp)
    """
    pattern = r"model\.\S+\.\S+(/version=\S+)?(/ckpt_id=\S+)?"
    if not re.match(pattern, uri):
        raise ValueError(f"[MOS] uri is invalid : {uri}")
    uri_partition = uri.split("/")
    level = "model"
    project, model_name = uri_partition[0].split(".")[1:]
    version = None
    ckpt_id = None
    if len(uri_partition) > 1:
        version = uri_partition[1].replace("version=", "")
        level = "version"
    if len(uri_partition) > 2:
        version = uri_partition[1].replace("version=", "")
        if version.find("max_pt") > 0:
            level = "version"
        else:
            ckpt_id = uri_partition[2].replace("ckpt_id=", "")
            level = "ckpt_id"
    # version will be automatically generated when auto_version is True
    if level == "model" and auto_version:
        version = time.strftime("%Y%m%d%H%M%S", time.localtime())
        uri = os.path.join(uri, f"version={version}")
    return dict(
        uri=uri,
        project=project,
        model_name=model_name,
        version=version,
        ckpt_id=ckpt_id,
        level=level,
    )


def render_uri_to_model_bank_path(uri, user_id):
    """Convert MOS URI to physical model bank path.

    This function resolves a MOS URI to its corresponding physical path in the
    model bank storage system. It handles version and checkpoint level URIs,
    with support for latest checkpoint resolution.

    Args:
        uri (str): MOS URI string (version or checkpoint level).
        user_id (str): User ID for MOS authentication.

    Returns:
        tuple: A tuple containing:
            - physical_path (str): Physical file system path to the model/checkpoint
            - full_uri (str): Complete URI including resolved checkpoint information

    Raises:
        ValueError: If URI is model-level only, version/checkpoint doesn't exist,
            or no checkpoints are found.

    Example:
        >>> user_id = "user123"
        >>> uri = "model.project.model_name/version=v1.0"
        >>> path, full_uri = render_uri_to_model_bank_path(uri, user_id)
        >>> print(path)  # '/path/to/model/storage/v1.0'
        >>>  specific checkpoint
        >>> uri = "model.project.model_name/version=v1.0/ckpt_id=step_1000"
        >>> path, full_uri = render_uri_to_model_bank_path(uri, user_id)
        >>> print(path)  # '/path/to/model/storage/v1.0/step_1000'
    """
    uri_info = parse_uri(uri, auto_version=False)
    full_uri = None
    physical_path = None
    if uri_info["level"] == "model":
        raise ValueError("[XDL_MOS] mos model_bank path not support only model")
    if uri_info["level"] == "version":
        version = ModelVersion(mos_version_uri=uri, user_id=user_id)
        if version.exists():
            version.query()
            if uri_info["version"] == "$max_pt":
                ckpt_list = version.get_latest_ckpt()
            else:
                ckpt_list = version.get_ckpt_list()
            logger.warning(
                f"[XDL_MOS] render_uri_to_model_bank_path uri={uri}, ckpt_list={ckpt_list}"
            )
            if len(ckpt_list) == 0:
                raise ValueError(f"[XDL_MOS] no ckpt found in version {uri}")
            last_ckpt = ckpt_list[0]
            full_uri = last_ckpt["uri"]
            if uri.find("ckpt_id") > 0:
                physical_path = os.path.join(
                    last_ckpt["real_physical_path"], last_ckpt["ckpt_id"]
                )
            else:
                physical_path = version.physical_path
        else:
            raise ValueError(f"[XDL_MOS] version not exist for {uri}")
    else:
        ckpt = ModelCkpt(uri=uri, user_id=user_id)
        if ckpt.exists():
            ckpt.query()
            physical_path = ckpt.physical_path
            full_uri = ckpt.uri
        else:
            raise ValueError(f"[XDL_MOS] ckpt not exists for {uri}")
    return physical_path, full_uri


def render_uri_to_output_dir(uri):
    """Convert MOS URI to output directory path for model saving.

    This function renders a MOS URI to an output directory path suitable for
    saving model checkpoints. It automatically generates a version if not present
    and creates or updates the corresponding MOS version entry.

    Args:
        uri (str): MOS URI string. If version is not specified, one will be
            automatically generated using current timestamp.

    Returns:
        str: Physical path to the output directory where models can be saved.

    Raises:
        ValueError: If URI specifies checkpoint ID (not allowed for output dirs)
            or if output directory cannot be obtained.

    Example:
        >>> # URI without version (auto-generated)
        >>> uri = "model.project.model_name"
        >>> output_dir = render_uri_to_output_dir(uri)
        >>> print(output_dir)  # '/path/to/output/20231201120000'
        >>> # URI with specific version
        >>> uri = "model.project.model_name/version=v1.0"
        >>> output_dir = render_uri_to_output_dir(uri)
        >>> print(output_dir)  # '/path/to/output/v1.0'

    Note:
        - Automatically retries MOS operations up to 3 times on failure
        - Uses CALCULATE_CLUSTER environment variable for cluster specification
        - Creates or updates MOS version entry as needed
    """
    uri_info = parse_uri(uri, auto_version=True)
    user_id = os.environ.get("USER_ID", None)
    if uri_info["level"] == "ckpt_id":
        # output dir should be a directory, so configuring ckpt id is not allowed
        raise ValueError("[MOS] nebula_mos not support ckpt_id")
    mos_obj = ModelVersion(mos_version_uri=uri_info["uri"], user_id=user_id)
    cluster = os.getenv("CALCULATE_CLUSTER", default=None)
    logger.info(f"[MOS] cluster = {cluster}")
    for index in range(3):
        try:
            mos_obj.create_or_update(cluster=cluster)
            break
        except Exception as e:
            if index < 2:
                time.sleep(5)
            else:
                raise e
    real_physical_path = mos_obj.physical_path
    if len(real_physical_path) == 0:
        raise ValueError(f"[MOS] can not get output_dir by uri : {uri}")
    return real_physical_path


class Mos:
    """MOS client wrapper for model management.

    This class provides a high-level interface for interacting with
    Mos system. It handles URI parsing, path resolution, and
    checkpoint management operations.

    The Mos class supports both model bank access (for loading existing models)
    and output directory creation (for saving new models), with automatic
    version management and checkpoint operations.

    Args:
        uri (str): MOS URI string specifying the model, version, and optionally checkpoint.
        model_bank_path (bool, optional): If True, resolve to model bank path for
            loading. If False, resolve to output directory for saving. Defaults to False.

    Attributes:
        uri (str): Original MOS URI
        uri_info (dict): Parsed URI components
        version_uri (str): Version-level URI for checkpoint operations
        user_id (str): User ID from USER_ID environment variable
        real_physical_path (str): Resolved physical file system path

    Raises:
        ValueError: If URI doesn't start with "model." or is otherwise invalid.

    Example:
        >>> import os
        >>> os.environ["USER_ID"] = "user123"
        >>> # Create MOS client for saving models
        >>> mos_saver = Mos("model.project.model_name/version=v1.0")
        >>> save_path = mos_saver.real_physical_path
        >>> print(f"Save models to: {save_path}")
        >>> # Update checkpoint after training
        >>> mos_saver.ckpt_update("epoch_10", "/local/path/to/checkpoint")
        >>> # Create MOS client for loading models
        >>> mos_loader = Mos(
        ...     "model.project.model_name/version=v1.0", model_bank_path=True
        ... )
        >>> load_path = mos_loader.real_physical_path
        >>> print(f"Load models from: {load_path}")
        >>> # Delete a checkpoint
        >>> mos_saver.ckpt_update("old_checkpoint", "", is_delete=True)

    Note:
        - Requires USER_ID environment variable to be set
        - Automatically handles version creation for output directories
        - Supports retry logic for robust checkpoint operations
    """

    def __init__(self, uri, model_bank_path=False):
        """Initialize MOS client with URI and path resolution mode.

        Args:
            uri (str): MOS URI string.
            model_bank_path (bool, optional): Path resolution mode. Defaults to False.

        Raises:
            ValueError: If URI format is invalid.
        """
        if not uri.startswith("model."):
            raise ValueError(f"[MOS] uri is invalid : {uri}")
        self.uri = uri
        self.uri_info = parse_uri(uri)
        self.version_uri = "model.{}.{}/version={}".format(
            self.uri_info["project"],
            self.uri_info["model_name"],
            self.uri_info["version"],
        )
        self.user_id = os.environ.get("USER_ID", None)
        if model_bank_path:
            self.real_physical_path, _ = render_uri_to_model_bank_path(
                uri, self.user_id
            )
        else:
            self.real_physical_path = render_uri_to_output_dir(uri)

    def ckpt_update(self, ckpt_id, path, is_delete=False):
        """Create, update, or delete a model checkpoint.

        This method manages individual checkpoints within a model version,
        supporting creation, updates, and deletion operations with automatic
        retry logic for reliability.

        Args:
            ckpt_id (str): Unique identifier for the checkpoint.
            path (str): Local file system path to the checkpoint data.
                Ignored when is_delete=True.
            is_delete (bool, optional): If True, delete the checkpoint.
                If False, create or update it. Defaults to False.

        Raises:
            Exception: If checkpoint operation fails after 3 retry attempts.

        Example:
            >>> mos_client = Mos("model.project.model_name/version=v1.0")
            >>> # Create/update checkpoint
            >>> mos_client.ckpt_update("epoch_5", "/path/to/checkpoint_epoch_5")
            >>> mos_client.ckpt_update("best_model", "/path/to/best_checkpoint")
            >>> # Delete checkpoint
            >>> mos_client.ckpt_update("old_checkpoint", "", is_delete=True)

        Note:
            - Automatically retries operations up to 3 times with 5-second delays
            - Logs checkpoint deletion operations as warnings
            - Uses the version_uri for checkpoint operations
        """
        ckpt = ModelCkpt(
            mos_version_uri=self.version_uri,
            ckpt_id=ckpt_id,
            user_id=self.user_id,
            path=path,
        )
        if is_delete:
            ckpt.delete()
            logger.warning(f"[MOS] delete ckpt_id = {ckpt_id}")
        else:
            for index in range(3):
                try:
                    ckpt.create_or_update()
                    break
                except Exception as e:
                    if index < 2:
                        time.sleep(5)
                    else:
                        raise e
