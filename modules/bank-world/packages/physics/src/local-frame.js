const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);

function radians(degrees) { return degrees * Math.PI / 180; }
function degrees(radiansValue) { return radiansValue * 180 / Math.PI; }

function geodeticToEcef({ lat, lon, altitude = 0 }) {
    const latitude = radians(lat);
    const longitude = radians(lon);
    const sinLat = Math.sin(latitude);
    const cosLat = Math.cos(latitude);
    const normal = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
    return {
        x: (normal + altitude) * cosLat * Math.cos(longitude),
        y: (normal + altitude) * cosLat * Math.sin(longitude),
        z: (normal * (1 - WGS84_E2) + altitude) * sinLat
    };
}

function ecefToGeodetic({ x, y, z }) {
    const lon = Math.atan2(y, x);
    const horizontal = Math.hypot(x, y);
    let lat = Math.atan2(z, horizontal * (1 - WGS84_E2));
    let altitude = 0;
    for (let iteration = 0; iteration < 8; iteration += 1) {
        const sinLat = Math.sin(lat);
        const normal = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
        altitude = horizontal / Math.cos(lat) - normal;
        lat = Math.atan2(z, horizontal * (1 - WGS84_E2 * normal / (normal + altitude)));
    }
    return { lat: degrees(lat), lon: degrees(lon), altitude };
}

class LocalGeodeticFrame {
    constructor(origin) {
        this.origin = { lat: origin.lat, lon: origin.lon, altitude: origin.altitude || 0 };
        this.ecefOrigin = geodeticToEcef(this.origin);
        const latitude = radians(this.origin.lat);
        const longitude = radians(this.origin.lon);
        this.east = [-Math.sin(longitude), Math.cos(longitude), 0];
        this.north = [
            -Math.sin(latitude) * Math.cos(longitude),
            -Math.sin(latitude) * Math.sin(longitude),
            Math.cos(latitude)
        ];
        this.up = [
            Math.cos(latitude) * Math.cos(longitude),
            Math.cos(latitude) * Math.sin(longitude),
            Math.sin(latitude)
        ];
    }

    toEnu(position) {
        const ecef = geodeticToEcef(position);
        const delta = [
            ecef.x - this.ecefOrigin.x,
            ecef.y - this.ecefOrigin.y,
            ecef.z - this.ecefOrigin.z
        ];
        const dot = (basis) => basis[0] * delta[0] + basis[1] * delta[1] + basis[2] * delta[2];
        return { east: dot(this.east), north: dot(this.north), up: dot(this.up) };
    }

    fromEnu({ east, north, up }) {
        const ecef = {
            x: this.ecefOrigin.x + this.east[0] * east + this.north[0] * north + this.up[0] * up,
            y: this.ecefOrigin.y + this.east[1] * east + this.north[1] * north + this.up[1] * up,
            z: this.ecefOrigin.z + this.east[2] * east + this.north[2] * north + this.up[2] * up
        };
        return ecefToGeodetic(ecef);
    }

    toRapier(position) {
        const enu = this.toEnu(position);
        return { x: enu.east, y: enu.up, z: -enu.north };
    }

    fromRapier({ x, y, z }) {
        return this.fromEnu({ east: x, north: -z, up: y });
    }
}

module.exports = { LocalGeodeticFrame, ecefToGeodetic, geodeticToEcef };
