---
name: uikit-ui-common-setup
description: Use when writing or refactoring UIKit UI in this repo, especially Workflow/ViewController wiring, Controller/View components, SnapKit layout, Workspace/Dflat-driven UI state, Objective-C adapter responders, or app threading boundaries.
---

# UIKit UI Common Setup

## First Pass

- Read nearby app code before introducing a new shape.
- Prefer existing Draw Things / Local Code UIKit patterns over new abstractions.
- Use `rg` / `rg --files` to find matching `Workflow`, `ViewController`, and `Controller` examples.

## Workflow And ViewController

- UI screens are managed by `Workflow` objects.
- A `Workflow` owns the concrete `UIViewController`.
- Expose `viewController` when the workflow needs the concrete type, and `viewControllable` as the generic UIKit-facing `UIViewController`.
- `UIViewController` subclasses are views plus UIKit event handling only.
- Do not put business logic, persistence policy, generation logic, networking, model work, or off-main-thread work in a `UIViewController`.
- `UIViewController` sends events out through delegates. The `Workflow` handles them and sets properties back on the view controller.

## Controller And View

- For shared business logic outside a workflow, use Controller & View.
- A controller is a plain Swift class, not `NSObject` and not a UIKit subclass.
- It owns one root `UIView` property, usually `let view: UIView`.
- Controllers are usually owned by workflows. Letting a view controller own one is only for lightweight, mostly animation-focused cases.

## Layout And View Setup

- Use lazy vars for referenced views. Each lazy var configures that view and self-contained subviews.
- Put hierarchy assembly and cross-view SnapKit constraints inline in `viewDidLoad` or the controller initializer.
- In `viewDidLoad` or the controller initializer, do layout first, then action/delegate hookups.
- Do not create one-off `configureXXX` / `setupXXX` methods for view setup, layout, or action wiring.
- Domain helpers are fine when they express behavior or reusable construction, such as `makeProjectMenu`, `updateSettingsSections`, or `didTapCollapseOrExpand`.

## One-Way Dataflow

- Persisted UI state should flow from Dflat / `Workspace` / `workspace.dictionary` observations into UI properties.
- Delegate callbacks should write source-of-truth state or call workflow/controller behavior, not directly mutate unrelated UI state.
- Even `didSet`-driven UI updates should be reached through the owning `Workflow` or real Controller when the state is not purely local view presentation.
- Keep view controllers as render targets: set properties on them; let them update UIKit controls.

## Objective-C And UIKit Adapters

- A `UIViewController` already inherits from `NSObject`; using it for UIKit lifecycle/delegates is fine.
- Plain controllers must stay plain Swift classes.
- When a plain controller needs Objective-C delegation, target/action, or UIKit delegate conformance, create a nested responder such as `Controller.ObjCResponder: NSObject`.
- The responder forwards callbacks back to the controller; it does not own business logic.

## Threads And Queues

- Do not use `async` / `await` in app UI code.
- `UIViewController` subclasses and views should only touch main-thread UIKit state.
- Workflow and Controller objects are the right places for off-main-thread behavior.
- Avoid creating private queues per workflow/controller. Reuse existing shared queues when behavior fits.
- Prefer one or two shared queues for broad work classes, like Draw Things' edit queue handling generation, tokenization, and related work.
- Create a new queue only when the behavior truly needs a separate serialization or QoS boundary. For rare fire-and-forget work, a global concurrent queue can be acceptable.
