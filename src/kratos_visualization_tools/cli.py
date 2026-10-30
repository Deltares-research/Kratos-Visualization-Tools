"""Command-line entry point for the MDPA visualizer."""
import argparse
import json
import os

from .project import find_mdpa_from_project
from .visualization import visualize_mdpa

def main():
    parser = argparse.ArgumentParser(description='Lightweight MDPA visualizer using matplotlib')
    parser.add_argument('mdpa', nargs='?', help='Path to mdpa file')
    parser.add_argument('--project', '-p', nargs='*', help='ProjectParameters.json file(s)')
    args = parser.parse_args()

    if not args.mdpa:
        if not args.project:
            parser.error('provide an mdpa file or --project with an ImportMDPAModeler')
        project_path = os.path.expanduser(args.project[0].strip("\"'"))
        if not os.path.isfile(project_path):
            parser.error(f"project file not found: {args.project[0]}")
        try:
            args.mdpa = find_mdpa_from_project(project_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if not os.path.isfile(args.mdpa):
            parser.error(f"mdpa file not found: {args.mdpa}")

    visualize_mdpa(args.mdpa, args.project or [])

