export default function ApprovalControls({
  disabled,
  onApprove,
  onReject,
}: {
  disabled: boolean;
  onApprove: () => void;
  onReject: () => void;
}) {
  return (
    <div className="approval">
      <button className="btn approve" disabled={disabled} onClick={onApprove}>
        ✓ Approve &amp; deliver
      </button>
      <button className="btn reject" disabled={disabled} onClick={onReject}>
        ✗ Reject
      </button>
    </div>
  );
}
