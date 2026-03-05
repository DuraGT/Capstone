"""
pose_fusion.py
--------------
Fuses marker detections from up to 3 cameras into one relative 6-DOF pose.

Rule: at least one camera must simultaneously see marker 0 (reference) AND
at least one of markers 1,2,3 (plane). Each qualifying camera produces an
independent estimate; estimates are averaged weighted by markers seen.
"""

import numpy as np
import cv2


class PoseFusion:
    def __init__(self):
        self.plane_ids = [1, 2, 3]
        self.ref_id    = 0

    def fuse(self, camera_results: list) -> tuple:
        estimates = []
        seen_ids  = set()
        for res in camera_results:
            rvecs = res.get('rvecs', {})
            tvecs = res.get('tvecs', {})
            seen_ids.update(rvecs.keys())
            pose, n = self._estimate(rvecs, tvecs)
            if pose is not None:
                estimates.append((pose, float(n)))
        if not estimates:
            return None, len(seen_ids)
        return self._weighted_avg(estimates), len(seen_ids)

    def _estimate(self, rvecs, tvecs):
        if self.ref_id not in rvecs:
            return None, 0
        visible = [p for p in self.plane_ids if p in rvecs]
        if not visible:
            return None, 0
        pos, euler = self._relative_pose(rvecs, tvecs, visible)
        if pos is None:
            return None, 0
        return np.array([*pos, *euler], dtype=np.float64), len(visible)

    def _relative_pose(self, rvecs, tvecs, plane_ids):
        try:
            R0, _ = cv2.Rodrigues(rvecs[self.ref_id])
            t0    = tvecs[self.ref_id].flatten()
            quats = [self._r2q(cv2.Rodrigues(rvecs[p])[0]) for p in plane_ids]
            Ravg  = self._q2r(self._avgq(quats))
            Rrel  = R0.T @ Ravg
            tavg  = np.mean([tvecs[p].flatten() for p in plane_ids], axis=0)
            trel  = R0.T @ (tavg - t0)
            return trel * 1000.0, self._euler(Rrel)
        except Exception:
            return None, None

    def _weighted_avg(self, estimates):
        poses   = np.array([e[0] for e in estimates])
        weights = np.array([e[1] for e in estimates])
        weights /= weights.sum()
        pos = np.average(poses[:, :3], axis=0, weights=weights)
        quats = [self._r2q(self._e2r(*p[3:])) for p in poses]
        q   = self._avgq(quats, weights)
        ang = self._euler(self._q2r(q))
        return np.array([*pos, *ang], dtype=np.float64)

    def _r2q(self, R):
        t = R[0,0]+R[1,1]+R[2,2]
        if t > 0:
            s = 0.5/np.sqrt(t+1)
            return np.array([0.25/s,(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s])
        elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
            return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
        elif R[1,1]>R[2,2]:
            s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
            return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
        else:
            s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
            return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])

    def _avgq(self, quats, weights=None):
        M = np.zeros((4,4))
        if weights is None: weights = np.ones(len(quats))/len(quats)
        for q,w in zip(quats,weights):
            q = q if q[0]>=0 else -q
            M += w*np.outer(q,q)
        return np.linalg.eigh(M)[1][:,-1]

    def _q2r(self, q):
        w,x,y,z = q/np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

    def _euler(self, R):
        sy = np.sqrt(R[0,0]**2+R[1,0]**2)
        if sy > 1e-6:
            return np.degrees([np.arctan2(R[2,1],R[2,2]),
                               np.arctan2(-R[2,0],sy),
                               np.arctan2(R[1,0],R[0,0])])
        return np.degrees([np.arctan2(-R[1,2],R[1,1]),
                           np.arctan2(-R[2,0],sy), 0.0])

    def _e2r(self, roll_deg, pitch_deg, yaw_deg):
        r,p,y = np.radians([roll_deg, pitch_deg, yaw_deg])
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        return Rz@Ry@Rx