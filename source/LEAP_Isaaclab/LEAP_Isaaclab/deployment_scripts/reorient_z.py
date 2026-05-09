#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
import argparse
from typing import Dict
from collections import deque
import traceback
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.leap_hand_utils.dynamixel_client import DynamixelClient
from utils.leap_hand_utils import leap_hand_utils as lhu
from load_model import RLGamesPolicy

class LEAPHandController:
    def __init__(
        self,
        cfg_path,
        ckpt_path,
        device="cuda:0",
        port="auto",
        baudrate=4000000,
        hz=30,
        kp=800.0,
        kd=200.0,
        curr_lim=500.0,
        max_steps=0,
        dry_run=False,
        disable_torque_on_exit=False,
    ):

        self.action_scale = 1 / 24
        self.action_type="relative"
        self.actions_num = 16
        self.hist_len = 3
        self.device = device
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.port = port
        self.baudrate = int(baudrate)
        self.hz = int(hz)
        self.control_dt = 1 / self.hz
        self.max_steps = int(max_steps)
        self.dry_run = dry_run
        self.disable_torque_on_exit = disable_torque_on_exit
        
        self.kP = float(kp)
        self.kI = 0.0
        self.kD = float(kd)
        self.curr_lim = float(curr_lim)
        self.ema_amount = 0.2
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()
        self.dxl_client = None
        if not self.dry_run:
            self.connect_hand()

    def connect_hand(self):
        candidate_ports = [self.port]
        if self.port == "auto":
            candidate_ports = [f"/dev/ttyUSB{idx}" for idx in range(4)] + [f"/dev/ttyACM{idx}" for idx in range(4)]

        last_error = None
        for port in candidate_ports:
            try:
                self.dxl_client = DynamixelClient(self.motors, port, self.baudrate)
                self.dxl_client.connect()
                print(f"Connected to LEAP hand on {port}")
                break
            except Exception as exc:
                last_error = exc
                self.dxl_client = None

        if self.dxl_client is None:
            raise RuntimeError(f"Failed to connect to LEAP hand. Last error: {last_error}")

        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.real_to_sim_indices]
        else:
            return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.sim_to_real_indices]
        else:
            return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = torch.tensor([ 4,  0,  8, 12,  6,  2, 10, 14,  7,  3, 11, 15,  1,  5,  9, 13], device=self.device)
        self.real_to_sim_indices= torch.tensor([ 1, 12,  5,  9,  0, 13,  4,  8,  2, 14,  6, 10,  3, 15,  7, 11], device=self.device)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        return sim_min, sim_max
    
    def LEAPsim_to_LEAPhand(self, joints):
        # joints = np.array(joints)
        ret_joints = joints + 3.14159
        return ret_joints

    def LEAPhand_to_LEAPsim(self, joints):
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        return joints

    def get_dof_limits(self):
        self.leap_dof_lower, self.leap_dof_upper = self.get_leap_hand_joint_limits()
        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)

    def get_leap_hand_joint_limits(self):
        upper_limits = [2.2300, 2.0940, 2.2300, 2.2300, 1.0470, 2.4430, 1.0470, 1.0470, 1.8850,
                       1.9000, 1.8850, 1.8850, 2.0420, 1.8800, 2.0420, 2.0420]
        lower_limits = [-0.3140, -0.3490, -0.3140, -0.3140, -1.0470, -0.4700, -1.0470, -1.0470,
                       -0.5060, -1.2000, -0.5060, -0.5060, -0.3660, -1.3400, -0.3660, -0.3660]
        return lower_limits, upper_limits

    def fetch_grasp_state(self):
        return torch.tensor([[0.000, 0.500, 0.000, 0.000, 
                             -0.750, 1.300, 0.000, 0.750, 
                              1.750, 1.500, 1.750, 1.750, 
                              0.00, 1.0000, 0.0000, 0.00]], device=self.device)

    def command_joint_position(self, desired_pose):
        if self.dry_run:
            self.curr_pos = desired_pose.clone()
            return
        desired_pose = self.LEAPsim_to_LEAPhand(desired_pose)
        # desired_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (self.leap_dof_upper - self.leap_dof_lower)
        desired_pose = self.sim_to_real(desired_pose) 
        desired_pose = desired_pose.detach().cpu().numpy().astype(float).flatten()
        
        # send command to motors
        self.dxl_client.write_desired_pos(self.motors, desired_pose)

    def poll_joint_position(self):
        if self.dry_run:
            return {'position': self.curr_pos.clone().squeeze(0)}
        # read position from hardware
        joint_position = self.dxl_client.read_pos()
        joint_position = torch.from_numpy(joint_position).to(device=self.device)
        
        joint_position = self.LEAPhand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower
        
        return {'position': joint_position}

    def deploy(self):
        print("Command to the initial position")
        warmup_steps = 1 if self.dry_run else self.hz * 4
        for _ in range(warmup_steps):
            self.command_joint_position(self.init_pose)
            robot_state = self.poll_joint_position()
            obses = robot_state['position']
            if not self.dry_run:
                time.sleep(self.control_dt)
        print("Initial position reached!")
       
        # Get current state
        self.command_joint_position(self.init_pose)
        robot_state = self.poll_joint_position()
        obses = robot_state['position']
        
        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)
        
        obs_hist_buf = torch.zeros((1, 32, self.hist_len), device=self.device, dtype=torch.double)
        prev_target = obses.clone()
        
        unscaled_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
        frame = torch.cat([unscaled_pos, prev_target], dim=-1).double()
        
        # Fill history buffer 
        for i in range(self.hist_len):
            obs_hist_buf[0, :, i] = frame
        obs_hist_buf[0, :, -1] = frame
        obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float() 

        counter = 0
        print("Starting policy execution:")
        try:
            while True:
                counter += 1
                start_time = time.time()

                # Get action from policy
                action = self.forward_network(obs_buf)
                action = action.squeeze(0)

                if self.action_type=="relative":
                    action = torch.clamp(action, -1.0, 1.0)
                    target = prev_target + self.action_scale * action
                elif self.action_type=="absolute":
                    action = unscale(action, self.leap_dof_lower, self.leap_dof_upper)
                    target = self.action_scale * action + (1.0 - self.action_scale) * prev_target 
                else:
                    raise ValueError(f"Unsupported action type: {self.action_type}. Must be relative or absolute.")

                target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
                prev_target = target.clone()
            
                print(f"Sending command: {target}")
                self.command_joint_position(target)
                # self.command_joint_position(self.init_pose)

                robot_state = self.poll_joint_position()
                print(f"Received state: {robot_state['position']}")

                obses = robot_state['position']
                unscaled_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)

                frame = torch.cat([unscaled_pos, target], dim=-1).double()
                obs_hist_buf[:, :, :-1] = obs_hist_buf[:, :, 1:]
                obs_hist_buf[:, :, -1] = frame
                obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float()

                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.control_dt - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if self.max_steps > 0 and counter >= self.max_steps:
                    print(f"Reached max steps: {self.max_steps}")
                    break
                    
        except KeyboardInterrupt:
            print("Stopping policy execution...")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            if self.dxl_client is not None and self.disable_torque_on_exit:
                self.dxl_client.set_torque_enabled(self.motors, False)
                print("Motors disabled.")

    def forward_network(self, obs):
        return self.player.step(obs)["selected_action"]
    
    def restore_policy(self):
        self.player = RLGamesPolicy(
            cfg_path=self.cfg_path,
            ckpt_path=self.ckpt_path,
            num_proprio_obs=96,
            action_space=self.actions_num,
            device=self.device,
        )
        self.player.reset_hidden_state()
        print("Model restored!")

    def manual_control(self, joint_positions):
        joint_positions = np.array(joint_positions)
        self.prev_pos = self.curr_pos
        self.curr_pos = joint_positions
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_hand_state(self):
        output = self.dxl_client.read_pos_vel_cur()
        return {
            'position': output[0].tolist(),
            'velocity': output[1].tolist(), 
            'effort': output[2].tolist()
        }
    
def unscale_np(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

def main():
    script_path = pathlib.Path(__file__).resolve()
    package_root = script_path.parents[1]
    project_root = script_path.parents[4]
    default_cfg = package_root / "tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
    default_checkpoint = project_root / "pretrained/leap_hand_reorient.pth"

    parser = argparse.ArgumentParser(description="Deploy a LEAP Hand z-axis reorientation policy.")
    parser.add_argument("--cfg", default=str(default_cfg), help="Path to the rl_games policy config.")
    parser.add_argument("--checkpoint", default=str(default_checkpoint), help="Path to the policy checkpoint.")
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--port", default="auto", help="Serial port, or auto.")
    parser.add_argument("--baudrate", type=int, default=4000000, help="Dynamixel baudrate.")
    parser.add_argument("--hz", type=int, default=30, help="Control frequency.")
    parser.add_argument("--kp", type=float, default=800.0, help="Motor P gain.")
    parser.add_argument("--kd", type=float, default=200.0, help="Motor D gain.")
    parser.add_argument("--curr-lim", type=float, default=500.0, help="Motor current limit in mA.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional deployment step limit.")
    parser.add_argument("--dry-run", action="store_true", help="Run policy inference without connecting to motors.")
    parser.add_argument("--disable-torque-on-exit", action="store_true", help="Disable torque when the process exits.")
    args = parser.parse_args()

    cfg_path = args.cfg
    ckpt_path = args.checkpoint
    
    controller = LEAPHandController(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        device=args.device,
        port=args.port,
        baudrate=args.baudrate,
        hz=args.hz,
        kp=args.kp,
        kd=args.kd,
        curr_lim=args.curr_lim,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        disable_torque_on_exit=args.disable_torque_on_exit,
    )
    controller.restore_policy()
    controller.deploy()

if __name__ == "__main__":
    main()
