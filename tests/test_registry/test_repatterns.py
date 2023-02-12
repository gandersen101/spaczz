"""Tests for pre-registered regex patterns."""
from catalogue import RegistryError
import pytest

from spaczz.registry.repatterns import re_patterns


def test_unregistered_pattern() -> None:
    """Raises `RegistryError`."""
    with pytest.raises(RegistryError):
        re_patterns.get("unregistered")


def test_dates() -> None:
    """Matches dates."""
    matching = ["1-19-14", "01-19-14", "1.19.14", "01.19.14", "1/19/14", "01/19/14"]
    pattern = re_patterns.get("dates")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_verbose_dates() -> None:
    """Matches verbose dates."""
    matching = ["January 19th, 2014", "Jan. 19th, 2014", "Jan 19 2014", "19 Jan 2014"]
    pattern = re_patterns.get("dates")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_times() -> None:
    """Matches times."""
    matching = ["09:45", "9:45", "23:45", "9:00am", "9am", "9:00 A.M.", "9:00 pm"]
    pattern = re_patterns.get("times")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_phones() -> None:
    """Matches phone numbers."""
    matching = [
        "12345678900",
        "1234567890",
        "+1 234 567 8900",
        "234-567-8900",
        "1-234-567-8900",
        "1.234.567.8900",
        "5678900",
        "567-8900",
        "(123) 456 7890",
        "+41 22 730 5989",
        "(+41) 22 730 5989",
        "+442345678900",
    ]
    pattern = re_patterns.get("phones")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_phones_with_extensions() -> None:
    """Matches phone numbers with extensions."""
    matching = [
        "(523)222-8888 ext 527",
        "(523)222-8888x623",
        "(523)222-8888 x623",
        "(523)222-8888 x 623",
        "(523)222-8888EXT623",
        "523-222-8888EXT623",
        "(523) 222-8888 x 623",
    ]
    non_matching = ["222-5555", "333-333-5555 dial 3"]
    pattern = re_patterns.get("phones_with_exts")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_links() -> None:
    """Matches web links."""
    matching = [
        "www.google.com",
        "http://www.google.com",
        "www.google.com/?query=dog",
        "sub.example.com",
        "http://www.google.com/%&#/?q=dog",
        "google.com",
    ]
    non_matching = ["www.google.con"]
    pattern = re_patterns.get("links")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_emails() -> None:
    """Matches emails."""
    matching = ["john.smith@gmail.com", "john_smith@gmail.com", "john@example.net"]
    non_matching = ["john.smith@gmail..com"]
    pattern = re_patterns.get("emails")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_ips() -> None:
    """Matches IP addresses."""
    matching = ["127.0.0.1", "192.168.1.1", "8.8.8.8"]
    pattern = re_patterns.get("ips")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_ipv6s() -> None:
    """Matches IPv6 addresses."""
    matching = [
        "fe80:0000:0000:0000:0204:61ff:fe9d:f156",
        "fe80:0:0:0:204:61ff:fe9d:f156",
        "fe80::204:61ff:fe9d:f156",
        "fe80:0000:0000:0000:0204:61ff:254.157.241.86",
        "fe80:0:0:0:0204:61ff:254.157.241.86",
        "fe80::204:61ff:254.157.241.86",
        "::1",
    ]
    pattern = re_patterns.get("ipv6s")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_prices() -> None:
    """Matches prices."""
    matching = ["$1.23", "$1", "$1,000", "$10,000.00"]
    non_matching = ["$1,10,0", "$100.000"]
    pattern = re_patterns.get("prices")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_hex_colors() -> None:
    """Matches hex colors."""
    matching = ["#fff", "#123", "#4e32ff", "#12345678"]
    pattern = re_patterns.get("hex_colors")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_credit_cards() -> None:
    """Matches credit cards."""
    matching = [
        "0000-0000-0000-0000",
        "0123456789012345",
        "0000 0000 0000 0000",
        "012345678901234",
    ]
    pattern = re_patterns.get("credit_cards")
    for m in matching:
        assert pattern.findall(m) == [m]


def test_btc_addresses() -> None:
    """Matches BTC addresses."""
    matching = [
        "1LgqButDNV2rVHe9DATt6WqD8tKZEKvaK2",
        "19P6EYhu6kZzRy9Au4wRRZVE8RemrxPbZP",
        "1bones8KbQge9euDn523z5wVhwkTP3uc1",
        "1Bow5EMqtDGV5n5xZVgdpRPJiiDK6XSjiC",
    ]
    non_matching = [
        "2LgqButDNV2rVHe9DATt6WqD8tKZEKvaK2",
        "19Ry9Au4wRRZVE8RemrxPbZP",
        "1bones8KbQge9euDn523z5wVhwkTP3uc12939",
        "1Bow5EMqtDGV5n5xZVgdpR",
    ]
    pattern = re_patterns.get("btc_addresses")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_street_addresses() -> None:
    """Matches street addresses."""
    matching = [
        "101 main st.",
        "504 parkwood drive",
        "3 elm boulevard",
        "500 elm street ",
    ]
    non_matching = ["101 main straight"]
    pattern = re_patterns.get("street_addresses")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_zip_codes() -> None:
    """Matches zip codes."""
    matching = ["02540", "02540-4119"]
    non_matching = ["101 main straight", "123456"]
    pattern = re_patterns.get("zip_codes")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_po_boxes() -> None:
    """Matches PO boxes."""
    matching = ["PO Box 123456", "p.o. box 234234"]
    non_matching = ["101 main straight"]
    pattern = re_patterns.get("po_boxes")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]


def test_ssn_number() -> None:
    """Matches SSN numbers."""
    matching = ["523 23 4566", "523-04-1234"]
    non_matching = ["774 00 1245", "666-12-7856"]
    pattern = re_patterns.get("ssn_numbers")
    for m in matching:
        assert pattern.findall(m) == [m]
    for m in non_matching:
        assert pattern.findall(m) != [m]
