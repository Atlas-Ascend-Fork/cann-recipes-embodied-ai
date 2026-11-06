# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run with: python3 -m unittest discover -s resources -p 'test_*.py'."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from download_assets import digest, download, install


class ResourceInstallerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.dest = self.root / 'workspace'

    def package(self, names=('vae/assets/hand.urdf',), symlink=False, expected=None):
        archive = self.root / 'test.tar.gz'
        content = b'<robot name="test"/>\n'
        files = {}
        with tarfile.open(archive, 'w:gz') as tar:
            for name in names:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/tmp/outside'
                    info.size = 0
                tar.addfile(info, None if symlink else io.BytesIO(content))
                files[name] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        return archive, {
            'size': archive.stat().st_size, 'sha256': digest(archive),
            'files': files if expected is None else expected,
        }

    def test_offline_and_idempotent(self):
        archive, spec = self.package()
        install(spec, archive, self.dest)
        install(spec, archive, self.dest)
        self.assertEqual((self.dest / 'vae/assets/hand.urdf').read_bytes(), b'<robot name="test"/>\n')

    def test_corrupt_archive_does_not_create_destination(self):
        archive, spec = self.package()
        archive.write_bytes(archive.read_bytes() + b'corrupt')
        with self.assertRaises(ValueError):
            install(spec, archive, self.dest)
        self.assertFalse(self.dest.exists())

    def test_different_existing_file_is_preserved(self):
        archive, spec = self.package()
        target = self.dest / 'vae/assets/hand.urdf'
        target.parent.mkdir(parents=True)
        target.write_text('user changes')
        with self.assertRaises(ValueError):
            install(spec, archive, self.dest)
        self.assertEqual(target.read_text(), 'user changes')

    def test_unsafe_or_duplicate_entries(self):
        cases = [(('../escape',), False), (('/absolute',), False),
                 (('link',), True), (('same', 'same'), False)]
        for names, symlink in cases:
            with self.subTest(names=names, symlink=symlink):
                archive, spec = self.package(names, symlink)
                with self.assertRaises(ValueError):
                    install(spec, archive, self.dest)
                self.assertEqual(list(self.dest.rglob('*')), [])

    def test_missing_and_unlisted_entries(self):
        archive, spec = self.package()
        spec['files']['missing'] = dict(next(iter(spec['files'].values())))
        with self.assertRaises(ValueError):
            install(spec, archive, self.dest)
        archive, spec = self.package(expected={})
        with self.assertRaises(ValueError):
            install(spec, archive, self.dest)
        self.assertEqual(list(self.dest.rglob('*')), [])

    def test_destination_symlink_is_rejected(self):
        archive, spec = self.package()
        outside = self.root / 'outside'
        outside.mkdir()
        self.dest.mkdir()
        (self.dest / 'vae').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            install(spec, archive, self.dest)
        self.assertEqual(list(outside.iterdir()), [])

    def test_late_bad_hash_leaves_destination_empty(self):
        archive, spec = self.package(('first', 'last'))
        spec['files']['last']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'File SHA256 mismatch'):
            install(spec, archive, self.dest)
        self.assertEqual(list(self.dest.rglob('*')), [])

    def test_conflict_does_not_install_other_files(self):
        archive, spec = self.package(('first', 'last'))
        self.dest.mkdir()
        (self.dest / 'last').write_text('user changes')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            install(spec, archive, self.dest)
        self.assertFalse((self.dest / 'first').exists())
        self.assertEqual((self.dest / 'last').read_text(), 'user changes')

    def test_non_directory_parent_is_rejected(self):
        archive, spec = self.package()
        self.dest.mkdir()
        (self.dest / 'vae').write_text('user file')
        with self.assertRaisesRegex(ValueError, 'Non-directory parent'):
            install(spec, archive, self.dest)
        self.assertEqual((self.dest / 'vae').read_text(), 'user file')

    def test_download_limits_size_and_rejects_insecure_redirect(self):
        cases = [('https://example.test/assets', b'extra', 2, 'exceeds'),
                 ('http://example.test/assets', b'data', 4, 'outside HTTPS')]
        for url, content, size, message in cases:
            with self.subTest(url=url), patch('download_assets.urllib.request.urlopen') as urlopen:
                stream = urlopen.return_value.__enter__.return_value
                stream.geturl.return_value = url
                stream.read.side_effect = io.BytesIO(content).read
                archive = self.root / 'download.tar.gz'
                with self.assertRaisesRegex(ValueError, message):
                    download('https://example.test/assets', archive, size)
                self.assertLessEqual(archive.stat().st_size, size)


if __name__ == '__main__':
    unittest.main()
