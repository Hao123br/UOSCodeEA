/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * NIST-developed software is provided by NIST as a public service. You may
 * use, copy and distribute copies of the software in any medium, provided that
 * you keep intact this entire notice. You may improve, modify and create
 * derivative works of the software or any portion of the software, and you may
 * copy and distribute such modifications or works. Modified works should carry
 * a notice stating that you changed the software and should note the date and
 * nature of any such change. Please explicitly acknowledge the National
 * Institute of Standards and Technology as the source of the software.
 * 
 * NIST-developed software is expressly provided "AS IS." NIST MAKES NO
 * WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF
 * LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
 * NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
 * UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST
 * DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE
 * SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
 * CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
 * 
 * You are solely responsible for determining the appropriateness of using and
 * distributing the software and you assume all risks associated with its use,
 * including but not limited to the risks and costs of program errors,
 * compliance with applicable laws, damage to or loss of data, programs or
 * equipment, and the unavailability or interruption of operation. This
 * software is not intended to be used in any situation where a failure could
 * cause risk of injury or damage to property. The software developed by NIST
 * employees is not subject to copyright protection within the United States.
 */

#include <string>

#include <ns3/buffer.h>
#include <ns3/log.h>
#include <ns3/object.h>
#include <ns3/type-id.h>

#include "mcptt-floor-machine-basic.h"
#include "mcptt-floor-msg-field.h"
#include "mcptt-msg.h"

#include "mcptt-floor-msg.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("McpttFloorMsg");

NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsg);

TypeId
McpttFloorMsg::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::McpttFloorMsg")
    .SetParent<McpttMsg> ()
  ;

  return tid;
}

McpttFloorMsg::McpttFloorMsg (void)
  : McpttMsg (),
    m_length (10),
    m_name ("MCPT"),
    m_padding (0),
    m_payloadType (204),
    m_ssrc (0),
    m_subtype (0),
    m_version (2)
{
  NS_LOG_FUNCTION (this);
}

McpttFloorMsg::McpttFloorMsg (uint8_t subtype, uint32_t ssrc)
  : McpttMsg (),
    m_length (10),
    m_name ("MCPT"),
    m_padding (0),
    m_payloadType (204),
    m_ssrc (ssrc),
    m_subtype (subtype),
    m_version (2)
{
  NS_LOG_FUNCTION (this << (uint32_t)subtype << ssrc);
}

McpttFloorMsg::McpttFloorMsg(const std::string& name, uint8_t payloadType, uint8_t subtype, uint32_t ssrc)
  : McpttMsg (),
    m_length (10),
    m_name (name),
    m_padding (0),
    m_payloadType (payloadType),
    m_ssrc (0),
    m_subtype (subtype),
    m_version (2)
{
  NS_LOG_FUNCTION (this << name << (uint32_t)payloadType << (uint32_t)subtype << ssrc);
}

McpttFloorMsg::~McpttFloorMsg (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsg::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldSsrc ssrcField;

  uint8_t firstByte = start.ReadU8 ();
  bytesRead += 1;
  // First two bits contain RTP version.
  uint8_t version = (firstByte >> 6);
  // Third bit is just a padding bit.
  uint8_t padding = (firstByte & 0x20) >> 5; 
  // The last five bits represents the message Subtype.
  uint8_t subtype = (firstByte & 0x1F);

  // The second byte represents the payload type (PT).
  uint8_t payloadType = start.ReadU8 ();
  bytesRead += 1;

  // The next two bytes describe the length of the message (not including the
  // four byte header that the length field is contained in).
  uint16_t length = start.ReadNtohU16 ();
  bytesRead += 2;

  // The next six bytes contain the synchronization source.
  bytesRead += ssrcField.Deserialize (start);
  uint32_t ssrc = ssrcField.GetSsrc ();

  // The next four bytes consist of four ascii characters which make up the
  // name.
  char nameChars[4];
  nameChars[0] = start.ReadU8 ();
  nameChars[1] = start.ReadU8 ();
  nameChars[2] = start.ReadU8 ();
  nameChars[3] = start.ReadU8 ();
  bytesRead += 4;

  std::string name (nameChars, 4);

  SetLength (length);
  SetName (name);
  SetPadding (padding);
  SetPayloadType (payloadType);
  SetSsrc (ssrc);
  SetSubtype (subtype);
  SetVersion (version);

  NS_LOG_LOGIC ("McpttFloorMsg read 14 bytes (length=" << (uint32_t)length << ";name=" << name << ";padding=" << (uint32_t)padding << ";payloadType=" << (uint32_t)payloadType << ";ssrc=" << ssrc << ";subtype=" << (uint32_t)subtype << ";version=" << (uint32_t)version << ";).");

  bytesRead += ReadData (start);

  // Read next 
  start.ReadNtohU32 ();  
  bytesRead += 4;
  start.ReadNtohU32 ();  
  bytesRead += 4;

  NS_LOG_LOGIC ("McpttFloorMsg read eight more bytes for security fields.");

  return bytesRead;
}

TypeId
McpttFloorMsg::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsg::GetTypeId ();
}

uint32_t
McpttFloorMsg::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);

  uint32_t size = 4; // 4 bytes for the first 4 bytes in the header.
  size += 8; // 8 more bytes for security fields in last part of the header.
  uint8_t length = GetLength ();

  size += length;

  NS_LOG_LOGIC ("McpttFloorMsg serialized size = " << size << " bytes.");

  return size;
}

void
McpttFloorMsg::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this);

  uint8_t version = GetVersion ();
  uint8_t padding = GetPadding ();
  uint8_t subtype = GetSubtype ();
  uint8_t payloadType = GetPayloadType ();
  uint16_t length = GetLength ();
  uint32_t ssrc = GetSsrc ();
  std::string name = GetName ();

  os << "RtcpHeader(";
  os << "V=" << (uint32_t)version << ";";
  os << "P=" << (uint32_t)padding << ";";
  os << "Subtype=" << (uint32_t)subtype << ";";
  os << "PT=" << (uint32_t)payloadType << ";";
  os << "length=" << (uint32_t)length << ";";
  os << "SSRC=" << (uint32_t)ssrc << ";";
  os << "name=" << name << ";";
  os << ")";
}

void
McpttFloorMsg::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("McpttRtcpHeader serializing...");

  uint8_t version = GetVersion ();
  uint8_t padding = GetPadding ();
  uint8_t subtype = GetSubtype ();
  uint8_t payloadType = GetPayloadType ();
  uint16_t length = GetLength ();
  uint32_t ssrc = GetSsrc ();
  std::string name = GetName ();
  
  McpttFloorMsgFieldSsrc ssrcField (ssrc);

  uint8_t firstByte = (version << 6) + (padding << 5) + subtype;

  start.WriteU8 (firstByte);
  start.WriteU8 (payloadType);
  start.WriteHtonU16 (length);
  ssrcField.Serialize (start);

  for (uint16_t index = 0; index < name.size (); index++)
    {
      uint8_t nameChar = name.at (index);

      start.WriteU8 (nameChar);
    }

  NS_LOG_LOGIC ("McpttFloorMsg wrote twelve bytes (length=" << (uint32_t)length << ";name=" << name << ";padding=" << (uint32_t)padding << ";payloadType=" << (uint32_t)payloadType << ";ssrc=" << ssrc << ";subtype=" << (uint32_t)subtype << ";version=" << (uint32_t)version << ";).");

  WriteData (start);

  // Write last 8 bytes for security.
  start.WriteHtonU32 (0);
  start.WriteHtonU32 (0);

  NS_LOG_LOGIC ("McpttFloorMsg wrote eight more bytes for security fields.");
}

void
McpttFloorMsg::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  NS_FATAL_ERROR ("Floor message not yet handled.");
}

uint32_t
McpttFloorMsg::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  return 0;
}

void
McpttFloorMsg::UpdateLength (const McpttFloorMsgField& currField, const McpttFloorMsgField& newField)
{
  NS_LOG_FUNCTION (this << currField << newField);

  uint8_t length = GetLength ();

  uint32_t oldSize = currField.GetSerializedSize ();
  uint32_t newSize = newField.GetSerializedSize ();

  NS_LOG_LOGIC ("McpttFloorMsg updating length from " << (uint32_t)length << " bytes to " << (length - oldSize + newSize) << " bytes.");

  length -= oldSize;
  length += newSize;

  SetLength (length);
}

void
McpttFloorMsg::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this << &buff);
}

uint16_t
McpttFloorMsg::GetLength (void) const
{
  NS_LOG_FUNCTION (this);

  return m_length;
}

std::string
McpttFloorMsg::GetName (void) const
{
  NS_LOG_FUNCTION (this);

  return m_name;
}

uint8_t
McpttFloorMsg::GetPadding (void) const
{
  NS_LOG_FUNCTION (this);

  return m_padding;
}

uint8_t
McpttFloorMsg::GetPayloadType (void) const
{
  NS_LOG_FUNCTION (this);

  return m_payloadType;
}

uint32_t
McpttFloorMsg::GetSsrc (void) const
{
  NS_LOG_FUNCTION (this);

  return m_ssrc;
}

uint8_t
McpttFloorMsg::GetSubtype (void) const
{
  NS_LOG_FUNCTION (this);

  return m_subtype;
}

uint8_t
McpttFloorMsg::GetVersion (void) const
{
  NS_LOG_FUNCTION (this);

  return m_version;
}

void
McpttFloorMsg::SetLength (uint16_t length)
{
  NS_LOG_FUNCTION (this << length);

  m_length = length;
}

void
McpttFloorMsg::SetName (const std::string& name)
{
  NS_LOG_FUNCTION (this << name);

  NS_ASSERT_MSG (name.size() == 4, "The name must contain 4 ASCII characters.");

  m_name = name;
}

void
McpttFloorMsg::SetPadding (uint8_t padding)
{
  NS_LOG_FUNCTION (this << padding);

  NS_ASSERT_MSG (padding < 2, "The padding bit must be either 0 or 1.");

  m_padding = padding;
}

void
McpttFloorMsg::SetPayloadType (uint8_t payloadType)
{
  NS_LOG_FUNCTION (this << payloadType);

  m_payloadType = payloadType;
}

void
McpttFloorMsg::SetSsrc (uint32_t ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  m_ssrc = ssrc;
}

void
McpttFloorMsg::SetSubtype (uint8_t subtype)
{
  NS_LOG_FUNCTION (this << subtype);

  m_subtype = subtype;
}

void
McpttFloorMsg::SetVersion (uint8_t version)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (version < 4, "The version can only range from 0 to 3.");

  m_version = version;
}
/** McpttFloorMsg - end **/

/** McpttFloorMsgRequest - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgRequest);

const uint8_t McpttFloorMsgRequest::SUBTYPE = 0;

TypeId
McpttFloorMsgRequest::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgRequest")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgRequest> ()
  ;

  return tid;
}

McpttFloorMsgRequest::McpttFloorMsgRequest (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgRequest::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint8_t length = GetLength ();
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldPriority priority;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  length += indicator.GetSerializedSize ();
  length += priority.GetSerializedSize ();
  length += trackInfo.GetSerializedSize ();
  length += userId.GetSerializedSize ();

  SetLength (length);
  SetIndicator (indicator);
  SetPriority (priority);
  SetTrackInfo (trackInfo);
  SetUserId (userId);
}

McpttFloorMsgRequest::~McpttFloorMsgRequest (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgRequest::ReadData (Buffer::Iterator& start)
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("McpttFloorMsgRequest deserializing...");

  McpttFloorMsg::ReadData (start);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldPriority priority;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  bytesRead += indicator.Deserialize (start);
  bytesRead += priority.Deserialize (start);
  bytesRead += trackInfo.Deserialize (start);
  bytesRead += userId.Deserialize (start);

  SetIndicator (indicator);
  SetPriority (priority);
  SetTrackInfo (trackInfo);
  SetUserId (userId);

  return bytesRead;
}

TypeId
McpttFloorMsgRequest::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgRequest::GetTypeId ();
}

void
McpttFloorMsgRequest::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this);

  os << "McpttFloorMsgRequest(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldIndic indicator = GetIndicator ();
  McpttFloorMsgFieldPriority priority = GetPriority ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldUserId userId = GetUserId ();

  indicator.Print (os);
  os << ";";
  priority.Print (os);
  os << ";";
  trackInfo.Print (os);
  os << ";";
  userId.Print (os);

  os << ")";
}

void
McpttFloorMsgRequest::WriteData (Buffer::Iterator& start) const
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("McpttFloorMsgRequest serializing...");

  McpttFloorMsg::WriteData (start);

  McpttFloorMsgFieldIndic indicator = GetIndicator ();
  McpttFloorMsgFieldPriority priority = GetPriority ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldUserId userId = GetUserId ();

  indicator.Serialize (start);
  priority.Serialize (start);
  trackInfo.Serialize (start);
  userId.Serialize (start);
}

void
McpttFloorMsgRequest::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgRequest::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorReq (*this);
}

McpttFloorMsgFieldIndic
McpttFloorMsgRequest::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

McpttFloorMsgFieldPriority
McpttFloorMsgRequest::GetPriority (void) const
{
  NS_LOG_FUNCTION (this);

  return m_priority;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgRequest::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgRequest::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

void
McpttFloorMsgRequest::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}

void
McpttFloorMsgRequest::SetPriority (const McpttFloorMsgFieldPriority& priority)
{
  NS_LOG_FUNCTION (this << priority);

  m_priority = priority;
}

void
McpttFloorMsgRequest::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgRequest::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}
/** McpttFloorMsgRequest - end **/

/** McpttFloorMsgGranted - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgGranted);

const uint8_t McpttFloorMsgGranted::SUBTYPE = 1;

TypeId
McpttFloorMsgGranted::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgGranted")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgGranted> ()
  ;

  return tid;
}

McpttFloorMsgGranted::McpttFloorMsgGranted (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgGranted::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this);

  uint8_t length = GetLength ();

  McpttFloorMsgFieldDuration duration;
  uint32_t grantedSsrc = 0;
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldPriority priority;
  McpttFloorMsgFieldQueueSize queueSize;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;
  McpttFloorMsgFieldSsrc ssrcField;
  std::list<McpttQueuedUserInfo> users;

  length += duration.GetSerializedSize ();
  length += ssrcField.GetSerializedSize ();
  length += indicator.GetSerializedSize ();
  length += priority.GetSerializedSize ();
  length += queueSize.GetSerializedSize ();
  length += userId.GetSerializedSize ();

  for (std::list<McpttQueuedUserInfo>::iterator it = users.begin (); it != users.end (); it++)
    {
      length += it->GetSerializedSize ();
    }

  length += trackInfo.GetSerializedSize ();

  SetLength (length);
  SetDuration (duration);
  SetGrantedSsrc (grantedSsrc);
  SetIndicator (indicator);
  SetPriority (priority);
  SetQueueSize (queueSize);
  SetTrackInfo (trackInfo);
  SetUserId (userId);
  SetUsers (users);
}

McpttFloorMsgGranted::~McpttFloorMsgGranted (void)
{
  NS_LOG_FUNCTION (this);
}

void
McpttFloorMsgGranted::AddUserInfo (const McpttQueuedUserInfo& userInfo)
{
  NS_LOG_FUNCTION (this << userInfo);

  uint8_t length = GetLength ();
  std::list<McpttQueuedUserInfo> users = GetUsers ();

  length += userInfo.GetSerializedSize ();

  users.push_back (userInfo);

  SetLength (length);
  SetUsers (users);
}

void
McpttFloorMsgGranted::ClearUserInfo (void)
{
  NS_LOG_FUNCTION (this);

  uint8_t length = GetLength ();
  std::list<McpttQueuedUserInfo> users = GetUsers ();

  for (std::list<McpttQueuedUserInfo>::iterator it = users.begin (); it != users.end (); it++)
    {
      length -= it->GetSerializedSize ();
    }

  users.clear ();

  SetLength (length);
  SetUsers (users);
}

uint32_t
McpttFloorMsgGranted::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_FUNCTION ("McpttFloorMsgGranted deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldDuration duration;
  uint32_t grantedSsrc = 0;
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldPriority priority;
  McpttFloorMsgFieldQueueSize queueSize;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;
  std::list<McpttQueuedUserInfo> users;

  bytesRead += duration.Deserialize (buff);

  grantedSsrc = buff.ReadNtohU32 ();
  bytesRead += 4;

  NS_LOG_LOGIC ("McpttFloorMsgGranted read four bytes (grantedSsrc=" << grantedSsrc << ").");

  bytesRead += priority.Deserialize (buff);
  bytesRead += userId.Deserialize (buff);
  bytesRead += queueSize.Deserialize (buff);

  for (uint16_t index = 0; index < queueSize.GetQueueSize (); index++)
    {
      McpttQueuedUserInfo userInfo;

      bytesRead += userInfo.Deserialize (buff);

      users.push_back (userInfo);
    }

  bytesRead += trackInfo.Deserialize (buff);

  SetDuration (duration);
  SetGrantedSsrc (grantedSsrc);
  SetPriority (priority);
  SetUserId (userId);
  SetQueueSize (queueSize);
  SetUsers (users);
  SetTrackInfo (trackInfo);

  return bytesRead;
}

TypeId
McpttFloorMsgGranted::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgGranted::GetTypeId ();
}

void
McpttFloorMsgGranted::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgGranted(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldDuration duration = GetDuration ();
  uint32_t grantedSsrc = GetGrantedSsrc ();
  McpttFloorMsgFieldPriority priority = GetPriority ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldQueueSize queueSize = GetQueueSize ();
  std::list<McpttQueuedUserInfo> users = GetUsers ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();

  duration.Print (os);
  os << ";";
  os << "grantedSsrc=" << grantedSsrc;
  os << ";";
  priority.Print (os);
  os << ";";
  userId.Print (os);
  os << ";";
  queueSize.Print (os);
  os << ";";
  os << "Users(";
  for (std::list<McpttQueuedUserInfo>::iterator it = users.begin (); it != users.end (); it++)
    {
      it->Print (os);
      os << ";";
    }
  os << ")";
  os << ";";
  trackInfo.Print (os);
  os << ")";
}

void
McpttFloorMsgGranted::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgGranted serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldDuration duration = GetDuration ();
  uint32_t grantedSsrc = GetGrantedSsrc ();
  McpttFloorMsgFieldPriority priority = GetPriority ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldQueueSize queueSize = GetQueueSize ();
  std::list<McpttQueuedUserInfo> users = GetUsers ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();

  duration.Serialize (buff);

  buff.WriteHtonU32 (grantedSsrc);

  NS_LOG_LOGIC ("McpttFloorMsgGranted wrote four bytes (grantedSsrc=" << grantedSsrc << ").");

  priority.Serialize (buff);
  userId.Serialize (buff);
  queueSize.Serialize (buff);

  for (std::list<McpttQueuedUserInfo>::iterator it = users.begin (); it != users.end (); it++)
    {
      it->Serialize (buff);
    }

  trackInfo.Serialize (buff);
}
 
void
McpttFloorMsgGranted::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgGranted::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorGranted (*this);
}

void
McpttFloorMsgGranted::UpdateUsers (const std::list<McpttQueuedUserInfo>& users)
{
  NS_LOG_FUNCTION (this << &users);

  ClearUserInfo ();

  for (std::list<McpttQueuedUserInfo>::const_iterator it = users.begin (); it != users.end (); it++)
    {
      AddUserInfo (*it);
    }
}

McpttFloorMsgFieldDuration
McpttFloorMsgGranted::GetDuration (void) const
{
  NS_LOG_FUNCTION (this);

  return m_duration;
}

uint32_t
McpttFloorMsgGranted::GetGrantedSsrc (void) const
{
  NS_LOG_FUNCTION (this);

  return m_grantedSsrc;
}

McpttFloorMsgFieldIndic
McpttFloorMsgGranted::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

McpttFloorMsgFieldPriority
McpttFloorMsgGranted::GetPriority (void) const
{
  NS_LOG_FUNCTION (this);

  return m_priority;
}

McpttFloorMsgFieldQueueSize
McpttFloorMsgGranted::GetQueueSize (void) const
{
  NS_LOG_FUNCTION (this);

  return m_queueSize;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgGranted::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgGranted::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

std::list<McpttQueuedUserInfo>
McpttFloorMsgGranted::GetUsers (void) const
{
  NS_LOG_FUNCTION (this);

  return m_users;
}

void
McpttFloorMsgGranted::SetDuration (const McpttFloorMsgFieldDuration& duration)
{
  NS_LOG_FUNCTION (this << duration);

  m_duration = duration;
}

void
McpttFloorMsgGranted::SetGrantedSsrc (uint32_t grantedSsrc)
{
  NS_LOG_FUNCTION (this << grantedSsrc);

  m_grantedSsrc = grantedSsrc;
}

void 
McpttFloorMsgGranted::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}

void
McpttFloorMsgGranted::SetPriority (const McpttFloorMsgFieldPriority& priority)
{
  NS_LOG_FUNCTION (this << priority);

  m_priority = priority;
}

void
McpttFloorMsgGranted::SetQueueSize (const McpttFloorMsgFieldQueueSize& queueSize)
{
  NS_LOG_FUNCTION (this << queueSize);

  m_queueSize = queueSize;
}

void
McpttFloorMsgGranted::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgGranted::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}

void
McpttFloorMsgGranted::SetUsers (const std::list<McpttQueuedUserInfo>&  users)
{
  NS_LOG_FUNCTION (this << &users);

  SetQueueSize (McpttFloorMsgFieldQueueSize (users.size ()));

  m_users = users;
}
/** McpttFloorMsgGranted - end **/

/** McpttFloorMsgDeny - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgDeny);

const uint8_t McpttFloorMsgDeny::SUBTYPE = 3;

TypeId
McpttFloorMsgDeny::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgDeny")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgDeny> ()
  ;

  return tid;
}

McpttFloorMsgDeny::McpttFloorMsgDeny (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgDeny::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint8_t length = GetLength ();
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldRejectCause rejCause;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  length += indicator.GetSerializedSize ();
  length += rejCause.GetSerializedSize ();
  length += trackInfo.GetSerializedSize ();
  length += userId.GetSerializedSize ();

  SetLength (length);
  SetRejCause (rejCause);
  SetTrackInfo (trackInfo);
  SetUserId (userId);
  SetIndicator (indicator);
}

McpttFloorMsgDeny::~McpttFloorMsgDeny (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgDeny::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgDeny deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldIndic indicator;
  McpttFloorMsgFieldRejectCause rejCause;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  bytesRead += rejCause.Deserialize (buff);
  bytesRead += userId.Deserialize (buff);
  bytesRead += trackInfo.Deserialize (buff);
  bytesRead += indicator.Deserialize (buff);

  SetRejCause (rejCause);
  SetUserId (userId);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);

  return bytesRead;
}

TypeId
McpttFloorMsgDeny::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgDeny::GetTypeId ();
}

void
McpttFloorMsgDeny::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgDeny(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldIndic indicator = GetIndicator ();
  McpttFloorMsgFieldRejectCause rejCause = GetRejCause ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();

  rejCause.Print (os);
  os << ";";
  userId.Print (os);
  os << ";";
  trackInfo.Print (os);
  os << ";";
  indicator.Print (os);

  os << ")";
} 

void
McpttFloorMsgDeny::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgDeny serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldIndic indicator = GetIndicator ();
  McpttFloorMsgFieldRejectCause rejCause = GetRejCause ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();

  rejCause.Serialize (buff);
  userId.Serialize (buff);
  trackInfo.Serialize (buff);
  indicator.Serialize (buff);
}
 
void
McpttFloorMsgDeny::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgDeny::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorDeny (*this);
}

McpttFloorMsgFieldIndic
McpttFloorMsgDeny::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

McpttFloorMsgFieldRejectCause
McpttFloorMsgDeny::GetRejCause (void) const
{
  NS_LOG_FUNCTION (this);

  return m_rejCause;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgDeny::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgDeny::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

void
McpttFloorMsgDeny::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}

void
McpttFloorMsgDeny::SetRejCause (const McpttFloorMsgFieldRejectCause& rejCause)
{
  NS_LOG_FUNCTION (this << rejCause);

  m_rejCause = rejCause;
}

void
McpttFloorMsgDeny::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgDeny::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}
/** McpttFloorMsgDeny - end **/

/** McpttFloorMsgRelease - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgRelease);

const uint8_t McpttFloorMsgRelease::SUBTYPE = 4;

TypeId
McpttFloorMsgRelease::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgRelease")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgRelease> ()
  ;

  return tid;
}

McpttFloorMsgRelease::McpttFloorMsgRelease (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgRelease::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint8_t length = GetLength ();

  McpttFloorMsgFieldUserId userId;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  length += userId.GetSerializedSize ();
  length += trackInfo.GetSerializedSize ();
  length += indicator.GetSerializedSize ();

  SetLength (length);
  SetUserId (userId);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);
}

McpttFloorMsgRelease::~McpttFloorMsgRelease (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgRelease::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgRelease deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldUserId userId;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  bytesRead += userId.Deserialize (buff);
  bytesRead += trackInfo.Deserialize (buff);
  bytesRead += indicator.Deserialize (buff);

  SetUserId (userId);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);

  return bytesRead;
}

TypeId
McpttFloorMsgRelease::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgRelease::GetTypeId ();
}

void
McpttFloorMsgRelease::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgRelease(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  userId.Print (os);
  os << ";";
  trackInfo.Print (os);
  os << ";";
  indicator.Print (os);

  os << ")";
}

void
McpttFloorMsgRelease::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgRelease serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  userId.Serialize (buff);
  trackInfo.Serialize (buff);
  indicator.Serialize (buff);
}

void
McpttFloorMsgRelease::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgRelease::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorRelease (*this);
}

McpttFloorMsgFieldIndic
McpttFloorMsgRelease::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgRelease::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgRelease::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

void
McpttFloorMsgRelease::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}

void
McpttFloorMsgRelease::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgRelease::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this);

  m_trackInfo = trackInfo;
}
/** McpttFloorMsgRelease - end **/

/** McpttFloorMsgTaken - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgTaken);

const uint8_t McpttFloorMsgTaken::SUBTYPE = 2;

TypeId
McpttFloorMsgTaken::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgTaken")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgTaken> ()
  ;

  return tid;
}

McpttFloorMsgTaken::McpttFloorMsgTaken (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgTaken::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint8_t length = GetLength ();

  McpttFloorMsgFieldGrantedPartyId partyId;
  McpttFloorMsgFieldPermToReq permission;
  McpttFloorMsgFieldUserId userId;
  McpttFloorMsgFieldSeqNum seqNum;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  length += partyId.GetSerializedSize ();
  length += permission.GetSerializedSize ();
  length += userId.GetSerializedSize ();
  length += seqNum.GetSerializedSize ();
  length += trackInfo.GetSerializedSize ();
  length += indicator.GetSerializedSize ();

  SetLength (length);
  SetPartyId (partyId);
  SetPermission (permission);
  SetUserId (userId);
  SetSeqNum (seqNum);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);
}

McpttFloorMsgTaken::~McpttFloorMsgTaken (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgTaken::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgTaken deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldGrantedPartyId partyId;
  McpttFloorMsgFieldPermToReq permission;
  McpttFloorMsgFieldUserId userId;
  McpttFloorMsgFieldSeqNum seqNum;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  bytesRead += partyId.Deserialize (buff);
  bytesRead += permission.Deserialize (buff);
  bytesRead += userId.Deserialize (buff);
  bytesRead += seqNum.Deserialize (buff);
  bytesRead += trackInfo.Deserialize (buff);
  bytesRead += indicator.Deserialize (buff);

  SetPartyId (partyId);
  SetPermission (permission);
  SetUserId (userId);
  SetSeqNum (seqNum);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);

  return bytesRead;
}

TypeId
McpttFloorMsgTaken::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgTaken::GetTypeId ();
}

void
McpttFloorMsgTaken::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgTaken(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldGrantedPartyId partyId = GetPartyId ();
  McpttFloorMsgFieldPermToReq permission = GetPermission ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldSeqNum seqNum = GetSeqNum ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  partyId.Print (os);
  permission.Print (os);
  userId.Print (os);
  seqNum.Print (os);
  trackInfo.Print (os);
  indicator.Print (os);

  os << ")";
}
  
void
McpttFloorMsgTaken::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("McpttFloorMsgTaken serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldGrantedPartyId partyId = GetPartyId ();
  McpttFloorMsgFieldPermToReq permission = GetPermission ();
  McpttFloorMsgFieldUserId userId = GetUserId ();
  McpttFloorMsgFieldSeqNum seqNum = GetSeqNum ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  partyId.Serialize (buff);
  permission.Serialize (buff);
  userId.Serialize (buff);
  seqNum.Serialize (buff);
  trackInfo.Serialize (buff);
  indicator.Serialize (buff);
}

void
McpttFloorMsgTaken::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgTaken::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorTaken (*this);
}

McpttFloorMsgFieldIndic
McpttFloorMsgTaken::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

McpttFloorMsgFieldGrantedPartyId
McpttFloorMsgTaken::GetPartyId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_partyId;
}

McpttFloorMsgFieldPermToReq
McpttFloorMsgTaken::GetPermission (void) const
{
  NS_LOG_FUNCTION (this);

  return m_permission;
}

McpttFloorMsgFieldSeqNum
McpttFloorMsgTaken::GetSeqNum (void) const
{
  NS_LOG_FUNCTION (this);

  return m_seqNum;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgTaken::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgTaken::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

void
McpttFloorMsgTaken::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}

void
McpttFloorMsgTaken::SetPartyId (const McpttFloorMsgFieldGrantedPartyId& partyId)
{
  NS_LOG_FUNCTION (this << partyId);

  m_partyId = partyId;
}

void
McpttFloorMsgTaken::SetPermission (const McpttFloorMsgFieldPermToReq& permission)
{
  NS_LOG_FUNCTION (this << permission);

  m_permission = permission;
}

void
McpttFloorMsgTaken::SetSeqNum (const McpttFloorMsgFieldSeqNum& seqNum)
{
  NS_LOG_FUNCTION (this << seqNum);

  m_seqNum = seqNum;
}

void
McpttFloorMsgTaken::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgTaken::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}
/** McpttFloorMsgTaken - end **/

/** McpttFloorMsgQueuePosReq - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgQueuePosReq);

const uint8_t McpttFloorMsgQueuePosReq::SUBTYPE = 8;

TypeId
McpttFloorMsgQueuePosReq::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgQueuePosReq")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgQueuePosReq> ()
  ;

  return tid;
}

McpttFloorMsgQueuePosReq::McpttFloorMsgQueuePosReq (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgQueuePosReq::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint32_t length = GetLength ();

  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  length += trackInfo.GetSerializedSize ();
  length += userId.GetSerializedSize ();

  SetLength (length);
  SetTrackInfo (trackInfo);
  SetUserId (userId);
}

McpttFloorMsgQueuePosReq::~McpttFloorMsgQueuePosReq (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgQueuePosReq::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorMsgQueuePosReq deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldUserId userId;

  bytesRead += userId.Deserialize (buff);
  bytesRead += trackInfo.Deserialize (buff);

  SetUserId (userId);
  SetTrackInfo (trackInfo);

  return bytesRead;
}

TypeId
McpttFloorMsgQueuePosReq::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgQueuePosReq::GetTypeId ();
}

void
McpttFloorMsgQueuePosReq::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgQueuePosReq(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldUserId userId = GetUserId ();

  userId.Print (os);
  trackInfo.Print (os);
  
  NS_LOG_LOGIC (")");
}

void
McpttFloorMsgQueuePosReq::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McptttFloorMsgQueuePosReq serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldUserId userId = GetUserId ();

  userId.Serialize (buff);
  trackInfo.Serialize (buff);
}

void
McpttFloorMsgQueuePosReq::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgQueuePosReq::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorQueuePositionReq (*this);
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgQueuePosReq::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldUserId
McpttFloorMsgQueuePosReq::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

void
McpttFloorMsgQueuePosReq::SetUserId (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgQueuePosReq::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}
/** McpttFloorMsgQueuePosReq - end **/

/** McpttFloorMsgQueueInfo - begin **/
NS_OBJECT_ENSURE_REGISTERED (McpttFloorMsgQueueInfo);

const uint8_t McpttFloorMsgQueueInfo::SUBTYPE = 9;

TypeId
McpttFloorMsgQueueInfo::GetTypeId (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  static TypeId tid = TypeId ("ns3::McpttFloorMsgQueueInfo")
    .SetParent<McpttFloorMsg> ()
    .AddConstructor<McpttFloorMsgQueueInfo> ()
  ;

  return tid;
}

McpttFloorMsgQueueInfo::McpttFloorMsgQueueInfo (uint32_t ssrc)
  : McpttFloorMsg (McpttFloorMsgQueueInfo::SUBTYPE, ssrc)
{
  NS_LOG_FUNCTION (this << ssrc);

  uint32_t length = GetLength ();

  McpttFloorMsgFieldUserId userId;
  uint32_t queuedSsrc = 1;
  McpttFloorMsgFieldSsrc queuedSsrcField;
  McpttFloorMsgFieldQueuedUserId queuedUserId;
  McpttFloorMsgFieldQueueInfo queueInfo;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  length += userId.GetSerializedSize ();
  length += queuedSsrcField.GetSerializedSize ();
  length += queuedUserId.GetSerializedSize ();
  length += queueInfo.GetSerializedSize ();
  length += trackInfo.GetSerializedSize ();
  length += indicator.GetSerializedSize ();

  SetLength (length);
  SetUserId (userId);
  SetQueuedSsrc (queuedSsrc);
  SetQueuedUserId (queuedUserId);
  SetQueueInfo (queueInfo);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);
}

McpttFloorMsgQueueInfo::~McpttFloorMsgQueueInfo (void)
{
  NS_LOG_FUNCTION (this);
}

uint32_t
McpttFloorMsgQueueInfo::ReadData (Buffer::Iterator& buff)
{
  NS_LOG_FUNCTION (this << &buff);

  NS_LOG_LOGIC ("McpttFloorQueueInfo deserializing...");

  McpttFloorMsg::ReadData (buff);

  uint32_t bytesRead = 0;
  McpttFloorMsgFieldUserId userId;
  uint32_t queuedSsrc = 1;
  McpttFloorMsgFieldSsrc queuedSsrcField;
  McpttFloorMsgFieldQueuedUserId queuedUserId;
  McpttFloorMsgFieldQueueInfo queueInfo;
  McpttFloorMsgFieldTrackInfo trackInfo;
  McpttFloorMsgFieldIndic indicator;

  bytesRead += userId.Deserialize (buff);

  bytesRead += queuedSsrcField.Deserialize (buff);
  queuedSsrc = queuedSsrcField.GetSsrc ();

  bytesRead += queuedUserId.Deserialize (buff);
  bytesRead += queueInfo.Deserialize (buff);
  bytesRead += trackInfo.Deserialize (buff);
  bytesRead += indicator.Deserialize (buff);

  SetUserId (userId);
  SetQueuedSsrc (queuedSsrc);
  SetQueuedUserId (queuedUserId);
  SetQueueInfo (queueInfo);
  SetTrackInfo (trackInfo);
  SetIndicator (indicator);

  return bytesRead;
}

TypeId
McpttFloorMsgQueueInfo::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);

  return McpttFloorMsgQueueInfo::GetTypeId ();
}

void
McpttFloorMsgQueueInfo::Print (std::ostream& os) const
{
  NS_LOG_FUNCTION (this << &os);

  os << "McpttFloorMsgQueueInfo(";

  McpttFloorMsg::Print (os);

  McpttFloorMsgFieldUserId userId = GetUserId ();
  uint32_t queuedSsrc = GetQueuedSsrc ();
  McpttFloorMsgFieldQueuedUserId queuedUserId = GetQueuedUserId ();
  McpttFloorMsgFieldQueueInfo queueInfo = GetQueueInfo ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  userId.Print (os);

  os << ";";
  os << "queuedSsrc=" << queuedSsrc;
  os << ";";
  queuedUserId.Print (os);
  os << ";";
  queueInfo.Print (os);
  os << ";";
  trackInfo.Print (os);
  os << ";";
  indicator.Print (os);

  os << ")";
}

void
McpttFloorMsgQueueInfo::WriteData (Buffer::Iterator& buff) const
{
  NS_LOG_FUNCTION (this);

  NS_LOG_LOGIC ("McpttFloorMsgQueueInfo serializing...");

  McpttFloorMsg::WriteData (buff);

  McpttFloorMsgFieldUserId userId = GetUserId ();
  uint32_t queuedSsrc = GetQueuedSsrc ();
  McpttFloorMsgFieldSsrc queuedSsrcField (queuedSsrc);
  McpttFloorMsgFieldQueuedUserId queuedUserId = GetQueuedUserId ();
  McpttFloorMsgFieldQueueInfo queueInfo = GetQueueInfo ();
  McpttFloorMsgFieldTrackInfo trackInfo = GetTrackInfo ();
  McpttFloorMsgFieldIndic indicator = GetIndicator ();

  userId.Serialize (buff);
  queuedSsrcField.Serialize (buff);
  queuedUserId.Serialize (buff);
  queueInfo.Serialize (buff);
  trackInfo.Serialize (buff);
  indicator.Serialize (buff);
}

void
McpttFloorMsgQueueInfo::UpdateTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  McpttFloorMsgFieldTrackInfo currTrackInfo = GetTrackInfo ();

  UpdateLength (currTrackInfo, trackInfo);

  SetTrackInfo (trackInfo);
}

void
McpttFloorMsgQueueInfo::Visit (McpttFloorMachineBasic& floorMachine) const
{
  NS_LOG_FUNCTION (this << &floorMachine);

  floorMachine.ReceiveFloorQueuePositionInfo (*this);
}

McpttFloorMsgFieldUserId
McpttFloorMsgQueueInfo::GetUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_userId;
}

uint32_t
McpttFloorMsgQueueInfo::GetQueuedSsrc (void) const
{
  NS_LOG_FUNCTION (this);

  return m_queuedSsrc;
}

McpttFloorMsgFieldQueuedUserId
McpttFloorMsgQueueInfo::GetQueuedUserId (void) const
{
  NS_LOG_FUNCTION (this);

  return m_queuedUserId;
}

McpttFloorMsgFieldQueueInfo
McpttFloorMsgQueueInfo::GetQueueInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_queueInfo;
}

McpttFloorMsgFieldTrackInfo
McpttFloorMsgQueueInfo::GetTrackInfo (void) const
{
  NS_LOG_FUNCTION (this);

  return m_trackInfo;
}

McpttFloorMsgFieldIndic
McpttFloorMsgQueueInfo::GetIndicator (void) const
{
  NS_LOG_FUNCTION (this);

  return m_indicator;
}

void
McpttFloorMsgQueueInfo::SetUserId  (const McpttFloorMsgFieldUserId& userId)
{
  NS_LOG_FUNCTION (this << userId);

  m_userId = userId;
}

void
McpttFloorMsgQueueInfo::SetQueuedSsrc (uint32_t queuedSsrc)
{
  NS_LOG_FUNCTION (this << queuedSsrc);

  m_queuedSsrc = queuedSsrc;
}

void
McpttFloorMsgQueueInfo::SetQueuedUserId (const McpttFloorMsgFieldQueuedUserId& queuedUserId)
{
  NS_LOG_FUNCTION (this << queuedUserId);

  m_queuedUserId = queuedUserId;
}

void
McpttFloorMsgQueueInfo::SetQueueInfo (const McpttFloorMsgFieldQueueInfo& queueInfo)
{
  NS_LOG_FUNCTION (this << queueInfo);

  m_queueInfo = queueInfo;
}

void
McpttFloorMsgQueueInfo::SetTrackInfo (const McpttFloorMsgFieldTrackInfo& trackInfo)
{
  NS_LOG_FUNCTION (this << trackInfo);

  m_trackInfo = trackInfo;
}

void
McpttFloorMsgQueueInfo::SetIndicator (const McpttFloorMsgFieldIndic& indicator)
{
  NS_LOG_FUNCTION (this << indicator);

  m_indicator = indicator;
}
/** McpttFloorMsgQueueInfo - end **/

} // namespace ns3
