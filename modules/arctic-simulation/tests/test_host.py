import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from atlantis_simulation.host import SimulationHost


class HostContractTests(unittest.TestCase):
    def test_explicit_database_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'ATLANTIS_SIM_DB_PATH'):
                SimulationHost().start()

    def test_network_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'loopback'):
            SimulationHost().start(host='0.0.0.0', database_path='/tmp/not-created.sqlite')

    def test_ipv6_url(self):
        host = SimulationHost()
        host._host = '::1'
        self.assertEqual(host.base_url, 'http://[::1]:5190')

    def test_start_stop_restart(self):
        host = SimulationHost()
        self.assertFalse(host.status()['running'])
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / 'scenario.sqlite'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            try:
                self.assertTrue(host.start(port=port, database_path=database)['started'])
                self.assertTrue(host.start(port=port, database_path=database)['alreadyRunning'])
                contract = host.command('GET', '/games/test/component-contract')
                self.assertIsInstance(contract, dict)
                self.assertTrue(host.stop()['stopped'])
                self.assertTrue(database.exists())
                self.assertTrue(host.start(port=port, database_path=database)['started'])
            finally:
                host.stop()
        self.assertFalse(host.status()['running'])
