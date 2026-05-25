import argparse

from config import get_config
from models.water_quality_model import build_model_from_config
from utils.data_loader import build_dataloaders
from utils.graph_utils import load_graphs
from utils.seed import set_seed
from utils.train_utils import attach_target_scaler, format_metrics, load_checkpoint, run_one_epoch


def parse_args():
    defaults = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=defaults.data_path)
    parser.add_argument("--adj-distance-path", default=defaults.adj_distance_path)
    parser.add_argument("--adj-hydrological-path", default=defaults.adj_hydrological_path)
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument("--device", default=defaults.device)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint_path, device=args.device)
    config = get_config()
    for key, value in checkpoint["config"].items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.data_path = args.data_path
    config.adj_distance_path = args.adj_distance_path
    config.adj_hydrological_path = args.adj_hydrological_path
    config.checkpoint_path = args.checkpoint_path
    config.device = args.device
    config.finalize()
    set_seed(config.seed)
    data = build_dataloaders(config)
    config.num_nodes = len(data.site_order)
    config.input_dim = len(data.feature_cols)
    config._target_scaler = data.target_scaler
    attach_target_scaler([data.train_loader, data.val_loader, data.test_loader], data.target_scaler)
    graphs = load_graphs(
        config.adj_distance_path,
        config.adj_hydrological_path,
        device=config.device,
        add_self_loops=config.add_self_loops,
    )
    model = build_model_from_config(config).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = run_one_epoch(model, data.test_loader, graphs, config, optimizer=None)
    print(f"test_loss={test_loss:.6f} {format_metrics('test', test_metrics)}")


if __name__ == "__main__":
    main()
