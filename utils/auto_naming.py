def get_exp_name(args):
    grd = ""
    grd += args.selection_method if args.selection_method != "none" else ""
    grd += f"-batchnummul{args.batch_num_mul}-interalmul{args.interval_mul}"           
    grd += f"_thresh-factor{args.check_thresh_factor}"
    folder = f"/{args.dataset}"
    args.save_dir += f"{folder}_{args.arch}_lr{args.lr}"
    if args.warm_start_epochs > 0:
        args.save_dir += f"_warm-{args.warm_start_epochs}"
    subset_size = args.train_frac
    args.save_dir += f"_train{subset_size:.2f}"
    if args.random_subset_size < 1.:
        args.save_dir += f"_random{args.random_subset_size:.2f}-start{args.partition_start}"
    grd += f'_dropevery{args.drop_interval}-loss{args.drop_thresh}-watch{args.watch_interval}' if args.drop_learned else ''
    args.save_dir += f"_batchsize{args.batch_size}_{grd}"
    if args.selection_method == 'crest':
        args.save_dir += f"_coreset" if args.approx_with_coreset else f"_subset"
        args.save_dir += f"_momentum" if args.approx_moment else f""

    args.save_dir += f'_seed_{args.seed}'

    return args.save_dir