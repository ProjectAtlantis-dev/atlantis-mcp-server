import {equipmentState,commandEquipment,stepEquipment} from './equipment-state.mjs';
function finite(value, fallback, minimum = -Infinity, maximum = Infinity) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.min(maximum, parsed)) : fallback;
}

function distanceKm(a, b) {
  const radians = degrees => degrees * Math.PI / 180;
  const latitudeDelta = radians(b.latitude - a.latitude);
  const longitudeDelta = radians(b.longitude - a.longitude);
  const latitudeA = radians(a.latitude);
  const latitudeB = radians(b.latitude);
  const haversine = Math.sin(latitudeDelta / 2) ** 2
    + Math.cos(latitudeA) * Math.cos(latitudeB) * Math.sin(longitudeDelta / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
}

/** Individual, server-authoritative support assets behind every defense site. */
export class VehicleLogisticsSystem {
  constructor({ onEvent = () => {} } = {}) {
    this.vehicles = new Map();
    this.onEvent = onEvent;
  }

  addSitePackage(site) {
    const roles = [
      ['radar', 'sensor'],
      ['command', 'command-and-control'],
      ['resupply', 'logistics'],
      ['recovery', 'maintenance-recovery'],
      ...(site.layers ?? []).map(layer => [`launcher-${layer.id}`, 'launcher']),
    ];
    for (const [suffix, role] of roles) {
      const bankAsset=site.bankAssets?.[suffix];
      const id = bankAsset?bankAsset.id:`${site.id}:${suffix}`;
      if (this.vehicles.has(id)) continue;
      const tankCapacityLiters = role === 'launcher' ? 500 : role === 'logistics' ? 900 : 650;
      const vehicle = {
        id,
        assetType: 'vehicle',
        ...(bankAsset?{bankAssetId:bankAsset.id,modelId:bankAsset.modelId,ownerAccountId:bankAsset.ownerAccountId,ownerUsername:bankAsset.ownerUsername,equipmentState:equipmentState(bankAsset.modelId)}:{}),
        siteId: site.id,
        role,
        position: { ...site.position },
        headingDeg: 0,
        status: site.readiness === 'deployed' ? 'deployed' : 'deploying',
        fuel: {
          liters: tankCapacityLiters,
          tankCapacityLiters,
          reserveLiters: tankCapacityLiters * 0.12,
          idleBurnLitersPerHour: role === 'sensor' ? 8 : 2.5,
        },
        condition: 1,
        operatingHours: 0,
        maintenanceIntervalHours: 72,
        recoveryRemainingSeconds: 0,
        losses: 0,
      };
      this.vehicles.set(id, vehicle);
      this.onEvent('vehicle-created', { vehicleId: id, siteId: site.id, role });
    }
  }

  reportPosition(input = {}) {
    const id = String(input.id ?? '').trim();
    if (!id) throw new Error('vehicle report requires id');
    const geodetic = {
      latitude: finite(input.latitude, NaN, -90, 90),
      longitude: finite(input.longitude, NaN, -180, 180),
      altitudeM: finite(input.altitudeM, 0),
    };
    if (!Number.isFinite(geodetic.latitude) || !Number.isFinite(geodetic.longitude)) {
      throw new Error('vehicle report requires valid latitude and longitude');
    }
    let vehicle = this.vehicles.get(id);
    if (vehicle == null) {
      const tankCapacityLiters = finite(input.tankCapacityLiters, 180, 1);
      vehicle = {
        id,
        assetType: 'vehicle',
        siteId: null,
        role: String(input.role ?? 'player-vehicle'),
        position: { x: 0, y: 0, z: geodetic.altitudeM },
        geodetic,
        headingDeg: finite(input.headingDeg, 0, 0, 360),
        status: 'deployed',
        fuel: {
          liters: tankCapacityLiters,
          tankCapacityLiters,
          reserveLiters: tankCapacityLiters * 0.12,
          idleBurnLitersPerHour: 1.5,
          burnLitersPerKm: finite(input.burnLitersPerKm, 0.5, 0),
        },
        condition: 1,
        operatingHours: 0,
        maintenanceIntervalHours: 72,
        recoveryRemainingSeconds: 0,
        losses: 0,
      };
      this.vehicles.set(id, vehicle);
      this.onEvent('vehicle-created', { vehicleId: id, siteId: null, role: vehicle.role });
    } else if (vehicle.geodetic != null) {
      const traveledKm = distanceKm(vehicle.geodetic, geodetic);
      vehicle.fuel.liters = Math.max(0, vehicle.fuel.liters - traveledKm * (vehicle.fuel.burnLitersPerKm ?? 0.5));
      vehicle.operatingHours += finite(input.elapsedSeconds, 0, 0, 3600) / 3600;
    }
    vehicle.geodetic = geodetic;
    vehicle.headingDeg = finite(input.headingDeg, vehicle.headingDeg, 0, 360);
    vehicle.lastReportedAt = finite(input.reportedAt, Date.now() / 1000, 0);
    this.onEvent('vehicle-position-reported', { vehicleId: id, geodetic: { ...geodetic }, headingDeg: vehicle.headingDeg });
    return structuredClone(vehicle);
  }

  setSiteReadiness(siteId, readiness) {
    for (const vehicle of this.vehicles.values()) {
      if (vehicle.siteId !== siteId || ['maintenance', 'recovering', 'lost'].includes(vehicle.status)) continue;
      vehicle.status = readiness === 'deployed' ? 'deployed' : 'deploying';
    }
  }

  syncEngagements(activeSiteIds) {
    for (const vehicle of this.vehicles.values()) {
      if (vehicle.role !== 'launcher' || ['maintenance', 'recovering', 'lost', 'deploying'].includes(vehicle.status)) continue;
      vehicle.status = activeSiteIds.has(vehicle.siteId) ? 'engaged' : 'deployed';
    }
  }

  commandEquipment(input){
    const vehicle=this.vehicles.get(input.id);
    if(!vehicle?.bankAssetId||!vehicle.equipmentState)throw Error('Unknown bank-owned site equipment');
    if(vehicle.ownerAccountId!==input.accountId)throw Error('Equipment owner required');
    commandEquipment(vehicle.modelId,vehicle.equipmentState,input);
    return structuredClone(vehicle);
  }

  step(dt) {
    const seconds = finite(dt, 0, 0, 1);
    for (const vehicle of this.vehicles.values()) {
      if(vehicle.equipmentState)stepEquipment(vehicle.modelId,vehicle.equipmentState,seconds);
      if (vehicle.status === 'lost' || vehicle.status === 'deploying') continue;
      if (vehicle.status === 'maintenance' || vehicle.status === 'recovering') {
        vehicle.recoveryRemainingSeconds = Math.max(0, vehicle.recoveryRemainingSeconds - seconds);
        if (vehicle.recoveryRemainingSeconds === 0) {
          vehicle.status = 'deployed';
          vehicle.fuel.liters = vehicle.fuel.tankCapacityLiters;
          vehicle.condition = 1;
          vehicle.operatingHours = 0;
          this.onEvent('vehicle-recovered', { vehicleId: vehicle.id, siteId: vehicle.siteId });
        }
        continue;
      }
      const engagementMultiplier = vehicle.status === 'engaged' ? 1.5 : 1;
      vehicle.operatingHours += seconds / 3600;
      vehicle.fuel.liters = Math.max(0, vehicle.fuel.liters - vehicle.fuel.idleBurnLitersPerHour * engagementMultiplier * seconds / 3600);
      vehicle.condition = Math.max(0, vehicle.condition - seconds / (vehicle.maintenanceIntervalHours * 3600) * 0.15);
      if (vehicle.fuel.liters <= vehicle.fuel.reserveLiters || vehicle.operatingHours >= vehicle.maintenanceIntervalHours) {
        vehicle.status = 'maintenance';
        vehicle.recoveryRemainingSeconds = 120;
        this.onEvent('vehicle-maintenance-started', {
          vehicleId: vehicle.id,
          siteId: vehicle.siteId,
          reason: vehicle.fuel.liters <= vehicle.fuel.reserveLiters ? 'fuel-reserve' : 'scheduled-service',
        });
      }
    }
  }

  counts() {
    const vehicles = [...this.vehicles.values()];
    const count = status => vehicles.filter(vehicle => vehicle.status === status).length;
    return {
      total: vehicles.length,
      available: count('available'),
      deployed: count('deployed'),
      engaged: count('engaged'),
      lost: count('lost'),
      deploying: count('deploying'),
      maintenance: count('maintenance'),
      recovering: count('recovering'),
    };
  }

  snapshot() {
    return structuredClone([...this.vehicles.values()]);
  }

  restore(vehicles) {
    if (!Array.isArray(vehicles)) throw new Error('vehicle state must be an array');
    const restored = new Map();
    for (const source of vehicles) {
      const id = String(source?.id ?? '').trim();
      if (!id) throw new Error('persisted vehicle is missing an id');
      if (restored.has(id)) throw new Error(`duplicate persisted vehicle id: ${id}`);
      const vehicle=structuredClone(source);
      if(vehicle.bankAssetId)vehicle.equipmentState=equipmentState(vehicle.modelId,vehicle.equipmentState);
      restored.set(id,vehicle);
    }
    this.vehicles = restored;
  }
}
