def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_unknown_route_404(client):
    res = client.get("/does-not-exist")
    assert res.status_code == 404
