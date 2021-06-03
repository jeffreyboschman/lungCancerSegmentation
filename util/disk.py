from diskcache import FanoutCache, Dis


def getCache(scope_str):
    return FanoutCache('data-unversioned/cache/' + scope_str,
                        disk=GzipDisk,
                        shards=64,
                        timeout=1,
                        size_limit=3e11,
                        # disk_min_file_size=2**20,
                        )
