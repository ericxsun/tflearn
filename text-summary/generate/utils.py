#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import logging

fmt = logging.Formatter(fmt="%(asctime)s %(filename)s:%(lineno)d: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(fmt)

LOGGER = logging.getLogger("text-summary-generate")
LOGGER.addHandler(handler)

LOGGER.setLevel(logging.DEBUG)
